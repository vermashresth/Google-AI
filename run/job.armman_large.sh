#data="armman_bootstrapped_eq_size"
data="armman_large_bootstrapped"
#data="armman_large_new"
save_string="setup_test"
N=80
B=100
seed=0
cdir="."
no_hawkins=1
S=2
max_epochs_double_oracle=10
agent_approach='combine_strategies'

# for running synthetic data
use_custom_data=0
n_arms=0
interval_size=-1

bash run_armman.sh ${cdir} ${seed} 0 ${data} ${save_string} ${N} ${B} ${no_hawkins} ${S} ${max_epochs_double_oracle} ${agent_approach} ${use_custom_data} ${n_arms} ${interval_size}

