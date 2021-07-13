#!/bin/sh

arr=(30 40 50 60)

for x in $arr; do
	i=1
	END=$x
	while [[ $i -le $END ]]
	do
		mod=$x%$i
		if [[ mod -eq 0 ]]
		then
			n=$(($x/$i))
			m_n="${i}_${n}"
			echo $m_n
			pkl_file_path="som_${m_n}"
			python rmab_individual_clustering_pilot.py ${m_n} som 2021-05-17 feb16-mar15_data/call/call_data_week_4.csv
			# echo $m_n
			echo
			echo
			python groundtruth_analysis.py ${m_n} som ${pkl_file_path}.pkl > "som_exps/${m_n}_rmse.txt"
			echo
			echo
			python get_soft_clusters.py 2021-05-17 feb16-mar15_data/call/call_data_week_4.csv ${pkl_file_path} rg/som_${m_n}_soft_whittle_indice.csv
		fi
		(( i=i+1 ))
	done
done