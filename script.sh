#!/bin/bash
conda activate tfnet

# Iterate through a list of items
for seed in 1; do
    for data in data20_101; do
        # python ./run_model.py --model u --d_id 3 --seed $seed --data $data 2>&1 | tee logs/${data}_${seed}_u.txt &
        # python run_model.py --model fno --d_id 5 --seed $seed --data $data 2>&1 | tee logs/${data}_${seed}_fno.txt &
        # python run_model.py --input_length 1 --model dhpm --d_id 5 --batch_size 6 --seed $seed --data $data 2>&1 | tee logs/${data}_${seed}_dhpm.txt &
        python run_model.py --model convlstm --d_id 6 --batch_size 14 --seed 1 --data $data 2>&1 | tee logs/${data}_${seed}_convlstm.txt &    
        python run_model.py --model convlstm --d_id 5 --batch_size 14 --seed 2 --data $data 2>&1 | tee logs/${data}_${seed}_convlstm.txt &    
        
        # python run_model.py --model gan --d_id 3 --seed $seed --data $data 2>&1 | tee logs/${data}_${seed}_gan.txt &
        # python run_model.py --model resnet --d_id 7 --batch_size 6 --seed $seed --data $data 2>&1 | tee logs/${data}_${seed}_dhpm.txt &
    done
    wait
done
wait