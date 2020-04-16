#/!bin/bash
num_patterns=(2 4 6 8 10 15 20 30 40)
adaptive=(0 1)
seeds=(0 1 2)
for num_p in "${num_patterns[@]}"
do
	for adapt in "${adaptive[@]}"
	do
		for seed in "${seeds[@]}"
		do
			export CUDA_VISIBLE_DEVICES=0
			python main.py --num_patterns $num_p --adaptive $adapt --random_seed $seed
		done
	done
done
