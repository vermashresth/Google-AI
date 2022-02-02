cd ..
python3 ${1}/double_oracle.py --hid 16 -l 2 --gamma 0.9 --cpu 1 \
--exp_name ${5} \
--home_dir ${1} \
-s ${2} \
--cannon ${3} \
--data ${4} \
--save_string ${5} \
-N ${6} -B ${7} \
--no_hawkins ${8} \
-S ${9} \
--horizon 10 \
--max_epochs_double_oracle ${10} \
--n_simu_epochs 10 \
--n_perturb 1 \
\
--agent_steps 10 \
--agent_epochs 1 \
--agent_approach ${11} \
\
--nature_steps 10 \
--nature_epochs 1 \
--gurobi_time_limit 5 \
\
--use_custom_data ${12} \
--n_arms ${13} \
--interval_size ${14}
