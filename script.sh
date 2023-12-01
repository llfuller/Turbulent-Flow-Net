#!/bin/bash
conda activate tfnet

seed=2

d_id=4
data=rbc_data
python run_model.py --data $data --d_id $d_id --kernel_size 3 --time_range 6 --output_length 6 --input_length 26 --batch_size 2 --num_epoch 100 --learning_rate 0.005 --decay_rate 0.9 --dropout_rate 0.0 --coef 0.001 --inp_dim 2 --seed $seed --model resnetmini 2>&1 | tee logs/${data}_${seed}_resnetmini.txt &

d_id=5
data=data9_101
python run_model.py --data $data --d_id $d_id --kernel_size 3 --time_range 6 --output_length 6 --input_length 26 --batch_size 2 --num_epoch 100 --learning_rate 0.005 --decay_rate 0.9 --dropout_rate 0.0 --coef 0.001 --inp_dim 2 --seed $seed --model resnetmini 2>&1 | tee logs/${data}_${seed}_resnetmini.txt &

d_id=6
data=data20_101
python run_model.py --data $data --d_id $d_id --kernel_size 3 --time_range 6 --output_length 6 --input_length 26 --batch_size 2 --num_epoch 100 --learning_rate 0.005 --decay_rate 0.9 --dropout_rate 0.0 --coef 0.001 --inp_dim 2 --seed $seed --model resnetmini 2>&1 | tee logs/${data}_${seed}_resnetmini.txt &

