data="armman"
save_string="ce_rmabppo_test"
N=10
B=5
robust_keyword="mid" # other option is "mid"
n_train_epochs=20
seed=0
cdir="."
no_hawkins=1

bash run/run_rmabppo_tiny_copy.sh ${cdir} ${seed} 0 ${data} ${save_string} ${N} ${B} ${robust_keyword} ${n_train_epochs} ${no_hawkins}



