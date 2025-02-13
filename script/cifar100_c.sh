beta=$1
method=$2
seed=$3
num_users=$4
echo $beta
# python -u main.py --group=c100t5 --exp_name=lander_b05 --dataset cifar100 --method=$method --tasks=5 --num_users 5 --beta=$beta --seed=$seed
# python -u main.py --group=c100t5 --exp_name=lander_b05 --dataset cifar100 --method=$method --tasks=10 --num_users 5 --beta=$beta --seed=$seed

python -u main.py --group=c100t10_$num_users --exp_name=$seed --dataset cifar100 --method=$method --tasks=10 --num_users $num_users --beta=$beta --seed=$seed --com_round 100





# python -u main.py --group=c100t10 --exp_name=0.1 --dataset cifar100 --method=$method --tasks=10 --num_users 5 --beta=$beta --seed=$seed --com_round 100
# CUDA_VISIBLE_DEVICES=2 python main.py --group=c100t5 --exp_name=lander_b05 --dataset cifar100 --method=$method --tasks=5 --num_users 5 --beta=$beta

# IID
# python main.py --group=c100t5 --exp_name=lander_b0 --dataset cifar100 --method=lander --tasks=5 --num_users 5 --beta=0

# NIID (beta=1)
# python main.py --group=c100t5 --exp_name=lander_b1 --dataset cifar100 --method=lander --tasks=5 --num_users 5 --beta=1

# NIID (beta=0.5)

# NIID (beta=0.1)
# python main.py --group=c100t5 --exp_name=lander_b01 --dataset cifar100 --method=lander --tasks=5 --num_users 5 --beta=0.1