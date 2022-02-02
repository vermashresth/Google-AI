data="armman_very_small"
# small, large
save_string="setup_test"
N=10
B=5
seed=0
cdir="."
no_hawkins=1
S=2
max_epochs_double_oracle=1
agent_approach='combine_strategies'

bash run_do_micro.sh ${cdir} ${seed} 0 ${data} ${save_string} ${N} ${B} ${no_hawkins} ${S} ${max_epochs_double_oracle} ${agent_approach} 0 0 -1



