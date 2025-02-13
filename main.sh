#!/bin/bash
set -e  
set -x  

tasks=$1
seed=$2
dataset=$3
beta=$4
com_round=$5
local_ep=$6
num_users=$7
net=$8
M=$9

echo $num_users
python -u main.py \
    --method STSA \
    --tasks $tasks \
    --dataset $dataset \
    --beta $beta \
    --com_round $com_round \
    --local_ep $local_ep \
    --num_users $num_users \
    --net $net \
    --M $M

