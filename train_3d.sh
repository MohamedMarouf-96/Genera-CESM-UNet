# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n A_un_3d --unregistered --dataset_mode A --d3 --batch-size 8 --gpu_ids 0,1,2,3
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n B_un_3d --unregistered --dataset_mode B --d3 --batch-size 8 --gpu_ids 0,1,2,3
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n B_un_3d --unregistered --dataset_mode B --d3 --batch-size 8 --gpu_ids 0,1,2,3 -l 0.000001 \
# --epochs 15
# EXPNAME=full3d
# for NAME in B
# do
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --batch-size 32 --gpu_ids 0,1,2,3 --epochs 1 \
# --input_channels 1  --dataset duke3d --d3 --batch-size 8 --balanced
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n  ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --eval_only --load ./checkpoint/${EXPNAME}_${NAME}/checkpoint_epoch1.pth \
# --input_channels 1 --dataset duke3d  --d3 --batch-size 8 --gpu_ids 0,1,2,3
# done
EXPNAME=full3d
for NAME in A B
do
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --batch-size 32 --gpu_ids 0,1,2,3 --epochs 5 \
# --input_channels 1  --dataset duke3d --d3 --batch-size 8
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n  ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --eval_only --load ./checkpoint/${EXPNAME}_${NAME}/checkpoint_epoch5.pth \
--input_channels 1 --dataset duke3d  --d3 --batch-size 4 --gpu_ids 0,1,2,3
exit
done

for NAME in ABD
do
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --batch-size 32 --gpu_ids 0,1,2,3 --epochs 5 \
# --input_channels 3  --dataset duke3d --d3 --batch-size 8
CUDA_VISIBLE_DEVICES=0 python train.py -n  ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --eval_only --load ./checkpoint/${EXPNAME}_${NAME}/checkpoint_epoch5.pth \
--input_channels 3 --dataset duke3d --d3
done

EXPNAME=full3d_balanced
for NAME in A B
do
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --batch-size 32 --gpu_ids 0,1,2,3 --epochs 5 \
# --input_channels 1  --dataset duke3d --d3 --balanced --batch-size 8
CUDA_VISIBLE_DEVICES=0 python train.py -n  ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --eval_only --load ./checkpoint/${EXPNAME}_${NAME}/checkpoint_epoch5.pth \
--input_channels 1 --dataset duke3d  --d3
done

for NAME in ABD
do
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --batch-size 32 --gpu_ids 0,1,2,3 --epochs 5 \
# --input_channels 3  --dataset duke3d --d3 --balanced --batch-size 8
CUDA_VISIBLE_DEVICES=0 python train.py -n  ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --eval_only --load ./checkpoint/${EXPNAME}_${NAME}/checkpoint_epoch5.pth \
--input_channels 3 --dataset duke3d --d3
done