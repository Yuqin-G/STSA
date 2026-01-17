# Enhancing Federated Class-Incremental Learning via Spatial-Temporal Statistics Aggregation (STSA)
## Introduction
This repository is the official implementation of the WWW 2026 paper "Enhancing Federated Class-Incremental Learning via Spatial-Temporal Statistics Aggregation".


## Usage

```python
# STSA
export CUDA_VISIBLE_DEVICES=0; tasks=10;
local_ep=2; com_round=10; num_users=5;
dataset="cifar224"; beta=0.5; 
net="vit_adapter"; M=1250
nohup sh ./main.sh "$tasks" "$seed" "$dataset" "$beta" "$com_round" "$local_ep" "$num_users" "$net" "$M" > test.log 2>&1 &
```
For STSA-E, set `args["type"] = 1` in main.py

## Acknowledgements
Our codebase is adapted from [LANDER](https://github.com/tmtuan1307/LANDER), [LAMDA-PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT). We thank the authors for their code!


## BibTeX & Citation
If you find this code useful, please consider citing our work:

```bibtex
@article{guan2025stsa,
  title={STSA: Federated Class-Incremental Learning via Spatial-Temporal Statistics Aggregation},
  author={Guan, Zenghao and Zhu, Guojun and Zhou, Yucan and Liu, Wu and Wang, Weiping and Luo, Jiebo and Gu, Xiaoyan},
  journal={arXiv preprint arXiv:2506.01327},
  year={2025}
}
```
