export CUDA_VISIBLE_DEVICES=1

python main2.py --anormly_ratio 0.85 --num_epochs 3   --batch_size 256  --mode train --dataset SMAP  --data_path SMAP --input_c 25    --output_c 25  --loss_fuc MSE --patch_size 357 --win_size 105
python main2.py --anormly_ratio 0.85  --num_epochs 10   --batch_size 256     --mode test    --dataset SMAP   --data_path SMAP  --input_c 25    --output_c 25   --loss_fuc MSE --patch_size 357 --win_size 105


python main2.py --anormly_ratio 0.6 --num_epochs 2   --batch_size 256  --mode train --dataset SMD  --data_path SMD   --input_c 38   --output_c 38  --loss_fuc MSE  --win_size 105  --patch_size 57
python main2.py --anormly_ratio 0.6 --num_epochs 10   --batch_size 256  --mode test    --dataset SMD   --data_path SMD     --input_c 38      --output_c 38   --loss_fuc MSE   --win_size 105  --patch_size 57