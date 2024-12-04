export CUDA_VISIBLE_DEVICES=1

python main2.py --anormly_ratio 1 --num_epochs 3   --batch_size 128  --mode train --dataset SWAT  --data_path SWAT  --input_c 51    --output_c 51  --loss_fuc MSE --patch_size 357 --win_size 105
python main2.py --anormly_ratio 1  --num_epochs 10   --batch_size 128     --mode test    --dataset SWAT   --data_path SWAT  --input_c 51    --output_c 51   --loss_fuc MSE --patch_size 357 --win_size 105

