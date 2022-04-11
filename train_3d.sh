# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n A_un_3d --unregistered --dataset_mode A --d3 --batch-size 8 --gpu_ids 0,1,2,3
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n B_un_3d --unregistered --dataset_mode B --d3 --batch-size 8 --gpu_ids 0,1,2,3
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n B_un_3d --unregistered --dataset_mode B --d3 --batch-size 8 --gpu_ids 0,1,2,3 -l 0.000001 \
--epochs 15