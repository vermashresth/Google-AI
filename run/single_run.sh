#!/bin/bash

# default values
#data="armman_bootstrapped_eq_size"  # no variance cluster size
#data="armman_bootstrapped_small_variance" # small variance cluster size
data="armman_large_bootstrapped"
#data="armman_bootstrapped_scale70"
#data="armman_bootstrapped_scale10"
#data="armman_bootstrapped_scale20"
#data="armman_large_new"
save_string="setup_test"
cdir="."
no_hawkins=1
S=2
max_epochs_double_oracle=10
agent_approach='combine_strategies'

seed=0
N=80 # num clusters * states
B=100 # budget
horizon=10 # horizon
# number of beneficiaries - n_arms
variation=1 # variation in cluster size

# for running synthetic data
# use_custom_data=1
# n_arms=1000000
# interval_size=-1 # variation in cluste rsize
use_custom_data=0
n_arms=0
interval_size=-1 # variation in cluste rsize

# for varying number of clusters
#N=20
N=80
#N=200
use_custom_data=1
n_arms=15320
interval_size="large" # variation in cluste rsize





while getopts s:n:b:h:i:I:a:c: option
do
case "${option}"
in
s) SEED=${OPTARG};;
n) N=${OPTARG};;
b) B=${OPTARG};;
h) horizon=${OPTARG};;
i) indiv=${OPTARG};;
I) interval_size=${OPTARG};;
a) n_arms=${OPTARG};;
c) use_custom_data=${OPTARG};;
esac
done

cd ..

for seed in {1..30}
do
#    cd ..
    python3 ${cdir}/double_oracle.py --hid 16 -l 2 --gamma 0.9 --cpu 1 \
    --exp_name ${save_string} \
    --home_dir ${cdir} \
    -s ${seed} \
    --cannon 0 \
    --data ${data} \
    --save_string ${save_string} \
    -N ${N} -B ${B} \
    --no_hawkins ${no_hawkins} \
    -S ${S} \
    --horizon ${horizon} \
    --variation ${variation} \
    --max_epochs_double_oracle ${max_epochs_double_oracle} \
    --n_simu_epochs 10 \
    --n_perturb 1 \
    \
    --agent_steps 10 \
    --agent_epochs 1 \
    --agent_approach ${agent_approach} \
    \
    --nature_steps 10 \
    --nature_epochs 1 \
    --gurobi_time_limit 5 \
    \
    --use_custom_data ${use_custom_data} \
    --n_arms ${n_arms} \
    --interval_size ${interval_size}

#    bash run_armman.sh ${cdir} ${seed} 0 ${data} ${save_string} ${N} ${B} ${no_hawkins} ${S} ${max_epochs_double_oracle} ${agent_approach} ${use_custom_data} ${n_arms} ${interval_size} ${horizon} ${variation}
done

