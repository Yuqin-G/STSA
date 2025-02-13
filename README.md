# STSAs

### Usage
```python
export CUDA_VISIBLE_DEVICES=0; tasks=10;
local_ep=2; com_round=10; num_users=5;
dataset="cifar224"; beta=0.5; 
net="vit_adapter"; M=1250
nohup sh ./main.sh "$tasks" "$seed" "$dataset" "$beta" "$com_round" "$local_ep" "$num_users" "$net" "$M" > test.log 2>&1 &
```