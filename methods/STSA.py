import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet, SimpleVitNet
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import copy, wandb
from sklearn.metrics import confusion_matrix
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def print_data_stats(client_id, train_data_loader):
    # pdb.set_trace()
    def sum_dict(a,b):
        temp = dict()
        # | 并集
        for key in a.keys() | b.keys():
            temp[key] = sum([d.get(key, 0) for d in (a, b)])
        return temp
    temp = dict()
    for batch_idx, (_, images, labels) in enumerate(train_data_loader):
        unq, unq_cnt = np.unique(labels, return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        temp = sum_dict(tmp, temp)
    print(sorted(temp.items(),key=lambda x:x[0]))


def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).cuda()
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits

def target2onehot(targets, n_classes):
    # onehot = torch.zeros(targets.shape[0], n_classes).cuda()
    onehot = torch.zeros(targets.shape[0], n_classes)
    # print(onehot.shape)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot

class STSA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if ("vit" in args['net']):
            self._network = SimpleVitNet(args, True)
        else:
            self._network = IncrementalNet(args, False)
        self.acc = []
        self.user_groups = None

    def after_task(self):
        self._known_classes = self._total_classes
        self.pre_loader = self.test_loader

    def setup_RP(self):
        M = self.args['M']
        self.M = M
        self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).cuda()).requires_grad_(False) # num classes in task x M
        self._network.RP_dim = M
        self.W_rand = torch.randn(self._network.fc.in_features, M).cuda()
        self._network.W_rand = self.W_rand

        self.P = torch.zeros(M, self.args["nb_classes"])
        self.mu = 0.0
        self.G = torch.zeros(M, M)
    
    def compute_multiple_sum_by_class(self, client_data, labels, num_means, num_classes):
        group_class_sums = []
        group_class_subset_sizes = []

        total_len = len(client_data)
        # torch.manual_seed(3)
        shuffled_indices = torch.randperm(total_len)
        step = total_len // num_means

        for i in range(num_means):
            start_idx = i * step
            end_idx = (i + 1) * step if i < num_means - 1 else total_len
            subset_indices = shuffled_indices[start_idx:end_idx]
            subset_data = client_data[subset_indices]
            subset_labels = labels[subset_indices]

            class_sum = [torch.zeros_like(client_data[0]) for _ in range(num_classes)]
            class_subset_size = [torch.tensor(0) for _ in range(num_classes)] 

            for data, label in zip(subset_data, subset_labels):
                class_sum[label] += data
                class_subset_size[label] += 1

            group_class_sums.append(class_sum)
            group_class_subset_sizes.append(class_subset_size)

        return group_class_sums, group_class_subset_sizes

    def replace_fc_ours3(self, trainloader, model, args, train_dataset):       
        user_groups = self.user_groups
        m = max(int(self.args["frac"] * self.args["num_users"]), 1)
        idxs_users = range(self.args["num_users"])
        model = model.eval()
        model.fc.W_rand=self.W_rand
        sum_list = []
        ss_list = []
        P_t = torch.zeros_like(self.P).double()
        G_t = torch.zeros_like(self.G).double()
        test_G = torch.zeros_like(self.G).double()
        L_t = []
        mu = 0
        
        for idx in idxs_users:
            e_t = []
            l_t = []
            local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                batch_size=self.args["local_bs"], shuffle=True, num_workers=8)
            with torch.no_grad():
                for i, batch in enumerate(local_train_loader):
                    (_, data, label) = batch
                    data = data.cuda()
                    label = label.cuda()
                    model = model.cuda()
                    embedding = model.extract_vector(data)
                    embedding = embedding.double()
                    e_t.append(embedding.cpu())
                    l_t.append(label.cpu())
                e_t = torch.cat(e_t, dim=0)
                l_t = torch.cat(l_t, dim=0)

                if not (args["M"] == 768 or args["M"] == 512):
                    e_t = F.relu(e_t @ self.W_rand.cpu().double())

                P_tt = e_t.T @ target2onehot(l_t, self.args["nb_classes"]).double()

                P_t += P_tt
                L_t += l_t

                self.P_t = P_t
                self.l_t = l_t
                self.L_t = L_t

                sum_multi, subset_size = self.compute_multiple_sum_by_class(e_t, l_t, args["dummy"], self.args["nb_classes"])
                for mm in sum_multi:
                    sum_list.append(torch.stack(mm))
                for ss in subset_size:
                    ss_list.append(torch.stack(ss))
        
        
        L_t = torch.stack(L_t)

        N = np.sum(ss_list)

        self.l_t = l_t 
        self.mu_list = sum_list
        self.ss_list = ss_list
        N_class = torch.tensor(np.sum(ss_list, axis=0))

        K = len(sum_list)
        c = self.args["nb_classes"]
        A = torch.where((N_class - 1) / (K - 1) < 0, torch.tensor(0.0), (N_class - 1) / (K - 1)).cuda()
        B = torch.where(((N_class - K)  / ((K - 1) * N_class)) < 0, torch.tensor(0.0), ((N_class - K) / ((K - 1) * N_class))).cuda()

        G_t = torch.zeros(self.args["M"], self.args["M"])
        G_t = G_t.cuda()
        for idx, s_t in enumerate(sum_list):
            s_t = s_t.double().cuda()
            for i in range(self.args["nb_classes"]):
                if (A[i] == 0): continue
                if (ss_list[idx][i] == 0): continue
                G_t += A[i] * torch.matmul(s_t[i].unsqueeze(1), s_t[i].unsqueeze(0)) / ss_list[idx][i]


        for i in range(self.args["nb_classes"]):
            if (B[i] == 0): continue
            G_t = G_t - B[i] * torch.matmul(P_t[:, i].unsqueeze(1), P_t[:, i].unsqueeze(0)).cuda()

        G_t = G_t.cpu()
        self.P = self.P + P_t
        self.G = self.G + G_t
        self.ridge = args["ridge"]

        # self.G = self.G.double()
        Wo = torch.linalg.solve(self.G + self.ridge*torch.eye(self.G.size(dim=0)), self.P).T  # better numerical stability than .invv
        Wo = Wo.float()

        if (args["M"] == 768 or args["M"] == 512):
            self._network.fc.W_rand = None
        self._network.fc.weight.data = Wo[0:self._network.fc.weight.shape[0],:].cuda()
        
        return model
    
    def replace_fc1(self, trainloader, model, args, train_dataset):       
        user_groups = self.user_groups
        m = max(int(self.args["frac"] * self.args["num_users"]), 1)
        model = model.eval()
        # print(model)
        model.fc.W_rand=self.W_rand
        model.fc.args=self.args
        idxs_users = range(self.args["num_users"])
        P_t = torch.zeros_like(self.P).double()
        G_t = torch.zeros_like(self.G).double()


        for idx in idxs_users:
            e_t = []
            l_t = []
            # print(idx, end = " ")
            local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                batch_size=self.args["local_bs"], shuffle=True, num_workers=8)
            with torch.no_grad():
                for i, batch in enumerate(local_train_loader):
                    (_, data, label) = batch
                    data = data.cuda()
                    label = label.cuda()
                    model = model.cuda()
                    embedding = model.extract_vector(data)

                    embedding = embedding.double()
                    e_t.append(embedding.cpu())
                    l_t.append(label.cpu())
                e_t = torch.cat(e_t, dim=0)
                l_t = torch.cat(l_t, dim=0)
                if not (args["M"] == 768 or args["M"] == 512):
                    e_t = F.relu(e_t @ self.W_rand.cpu().double())

                P_t += e_t.T @ target2onehot(l_t, self.args["nb_classes"]).double()
                G_t += e_t.T @ e_t       


        self.P = self.P + P_t
        self.G = self.G + G_t
        self.ridge = args["ridge"]
        Wo = torch.linalg.solve(self.G + self.ridge*torch.eye(self.G.size(dim=0)), self.P).T  # better numerical stability than .invv
        Wo = Wo.float() 

        if (args["M"] == 768 or args["M"] == 512):
            self._network.fc.W_rand = None
        self._network.fc.weight.data = Wo[0:self._network.fc.weight.shape[0],:].cuda()
        return model
    
    def incremental_train(self, data_manager):
        self.args["nb_classes"] = data_manager.nb_classes
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        # del self._network.fc
        # self._network.fc=None
        self._network.update_fc(self._total_classes)
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self.shot = None
        # print(self.shot)

        train_dataset = data_manager.get_dataset(   #* get the data for one task
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train", shot=self.shot
        )
        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test", shot=self.shot)
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.args["local_bs"], shuffle=True, num_workers=8)
        
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=8
        )

        setup_seed(self.seed)
    
        if self._cur_task == 0:
            self._network.fc.W_rand = None
            self._fl_train(train_dataset, self.test_loader)
        if (self._cur_task == 0):
            self.setup_RP()
        if self._cur_task > 0:
            # user_groups, ds = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"], shot=self.shot, eta=self.args['eta'])
            user_groups, ds = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
            # print(ds)
            self.user_groups = user_groups
        if (self.args["type"] == 0):
            self.replace_fc1(self.train_loader_for_protonet, self._network, self.args, train_dataset_for_protonet)
        elif (self.args["type"] == 1):
            self.replace_fc_ours3(self.train_loader_for_protonet, self._network, self.args, train_dataset_for_protonet)

    def _local_update(self, model, train_data_loader, lr):
        # print(lr)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for iter in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                loss = F.cross_entropy(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def per_cls_acc(self, val_loader, model):
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for i, (_, input, target) in enumerate(val_loader):
                input, target = input.cuda(), target.cuda()
                # compute output
                output = model(input)["logits"]
                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        cf = confusion_matrix(all_targets, all_preds).astype(float)

        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)

        cls_acc = cls_hit / cls_cnt
        return cls_acc
        # pdb.set_trace()
        # out_cls_acc = 'Per Class Accuracy: %s' % ((np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        # print(out_cls_acc)
        
    def _local_finetune(self, model, train_data_loader):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # print_data_stats(0, train_data_loader)
        for iter in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                fake_targets = labels - self._known_classes
                output = model(images)["logits"]
                #* finetune on the new tasks
                loss = F.cross_entropy(output[:, self._known_classes :], fake_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # self.per_cls_acc(self.test_loader, model)

        return model.state_dict()

    def _fl_train(self, train_dataset, test_loader):
        self._network.cuda()
        cls_acc_list = []
        # user_groups, ds = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"], shot=self.shot, eta=self.args["eta"])
        user_groups, ds = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        print(ds)
        # return
        self.user_groups = user_groups
        prog_bar = tqdm(range(self.args["com_round"]), total=self.args["com_round"])
        # for _, com in enumerate(prog_bar):
        optimizer = torch.optim.SGD(self._network.parameters(), lr=self.args['local_lr'], momentum=0.9, weight_decay=self.args['weight_decay'])
        if self.args["dataset"] == "tiny_imagenet":
            print("MultiStepLR")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args["com_round"], eta_min=0)
        for _, com in enumerate(range(self.args["com_round"])):
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                print(idx, end = " ")
                # print(idx)
                local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                    batch_size=self.args["local_bs"], shuffle=True, num_workers=8, pin_memory=True)
                # local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                    # batch_size=self.args["local_bs"], shuffle=True, num_workers=0)
                # print(scheduler.get_last_lr()[0])
                if self._cur_task == 0:
                    w = self._local_update(copy.deepcopy(self._network), local_train_loader, scheduler.get_last_lr()[0])
                    netw = copy.deepcopy(self._network)
                    netw.load_state_dict(w)
                    test_acc = self._compute_accuracy(netw, test_loader)
                    print(test_acc)
                else:
                    w = self._local_finetune(copy.deepcopy(self._network), local_train_loader)
                local_weights.append(copy.deepcopy(w))
            # update global weights

            scheduler.step()
            global_weights = average_weights(local_weights)
            # self._network.load_state_dict(weight)
            self._network.load_state_dict(global_weights)
            if com % 1 == 0:
                cls_acc = self.per_cls_acc(self.test_loader, self._network)
                cls_acc_list.append(cls_acc)

                test_acc = self._compute_accuracy(self._network, test_loader)
                info=("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                    self._cur_task, com + 1, self.args["com_round"], test_acc,))
                prog_bar.set_description(info)

        if (self.args["com_round"] == 0):
            global_weights = torch.load(self.args["file"])
            self._network.load_state_dict(global_weights)
            print(self._network.fc)
            print("========= load ==========")
            cls_acc = self.per_cls_acc(self.test_loader, self._network)
            cls_acc_list.append(cls_acc)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info=("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                0, 0, self.args["com_round"], test_acc,))
            prog_bar.set_description(info)

        if (self.args["com_round"] > 0):
            # torch.save(global_weights, f'ours_{self.args["dataset"]}_{self.args["num_users"]}_{self.args["net"]}_{self.args["beta"]}_{self.args["seed"]}' + '.pkl')
            # torch.save(global_weights, f'new_ours_{self.args["dataset"]}_{self.args["num_users"]}_{self.args["net"]}_{self.args["beta"]}_{self.args["seed"]}_com_{self.args["com_round"]}' + '.pkl')
            torch.save(global_weights, f'e_new_ours_{self.args["dataset"]}_{self.args["num_users"]}_{self.args["net"]}_{self.args["beta"]}_{self.args["seed"]}_com_{self.args["com_round"]}' + '.pkl')
        acc_arr = np.array(cls_acc_list)
        acc_max = acc_arr.max(axis=0)
        if self._cur_task == 4:
            acc_max = self.per_cls_acc(self.test_loader, self._network)
        # print(" ")
        # print("For task: {}, acc list max: {}".format(self._cur_task, acc_max))
        self.acc.append(acc_max)



