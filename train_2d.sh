# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n A_with_conf --dataset_mode A --batch-size 16 --gpu_ids 0,1,2,3
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n A_with_conf --dataset_mode A --batch-size 16 --gpu_ids 0,1,2,3
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n A_with_conf --dataset_mode A --batch-size 16 --gpu_ids 0,1,2,3
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n A_with_conf --dataset_mode A --batch-size 16 --gpu_ids 0,1,2,3
# for NAME in A B ABD
# do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n ${NAME}_with_conf_freeze --dataset_mode ${NAME} --batch-size 16 --gpu_ids 0,1,2,3 --load ./checkpoint/${NAME}/checkpoint_epoch5.pth --epochs 1
#     CUDA_VISIBLE_DEVICES=0 python train.py -n ${NAME}_with_conf_freeze --dataset_mode ${NAME} --eval_only --load ./checkpoint/${NAME}_with_conf_freeze/checkpoint_epoch1.pth
# done
# for NAME in A ABD
# do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n ${NAME}_with_conf_freeze_un --unregistered --dataset_mode ${NAME} --batch-size 16 --gpu_ids 0,1,2,3 --load ./checkpoint/${NAME}_un/checkpoint_epoch5.pth --epochs 1
#     CUDA_VISIBLE_DEVICES=0 python train.py -n ${NAME}_with_conf_freeze_un --unregistered --dataset_mode ${NAME} --eval_only --load ./checkpoint/${NAME}_with_conf_freeze_un/checkpoint_epoch1.pth
# done
# NAME=B
# CUDA_VISIBLE_DEVICES=1 python train.py -n $NAME --dataset_mode B --eval_only --load ./checkpoint/$NAME/checkpoint_epoch5.pth
# NAME=AA
# CUDA_VISIBLE_DEVICES=0 python train.py -n $NAME --dataset_mode $NAME --eval_only --load ./checkpoint/$NAME/checkpoint_epoch5.pth
# NAME=A
# CUDA_VISIBLE_DEVICES=0 python train.py -n  full2d_balanced_debug${NAME} --unregistered --dataset_mode ${NAME} --load ./checkpoint/full2d_balanced_${NAME}/checkpoint_epoch5.pth \
# --input_channels 1 --eval_only --dataset duke2d --batch-size 32
EXPNAME=full2d
for NAME in A B
do
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --batch-size 32 --gpu_ids 0,1,2,3 --epochs 5 \
--input_channels 1  --dataset duke2d
CUDA_VISIBLE_DEVICES=0 python train.py -n  ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --eval_only --load ./checkpoint/${EXPNAME}_${NAME}/checkpoint_epoch5.pth \
--input_channels 1 --dataset duke2d #--batch-size 32
done

for NAME in ABD
do
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --batch-size 32 --gpu_ids 0,1,2,3 --epochs 5 \
--input_channels 3  --dataset duke2d
CUDA_VISIBLE_DEVICES=0 python train.py -n  ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --eval_only --load ./checkpoint/${EXPNAME}_${NAME}/checkpoint_epoch5.pth \
--input_channels 3 --dataset duke2d #--batch-size 32
done
# for NAME in A
# do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n dn${NAME}_blocking --unregistered --dataset_mode dn${NAME} --batch-size 16 --gpu_ids 0,1,2,3 --epochs 5
#     CUDA_VISIBLE_DEVICES=0 python train.py -n dn${NAME}_blocking --unregistered --dataset_mode dn${NAME} --eval_only --load ./checkpoint/dn${NAME}_blocking/checkpoint_epoch5.pth
# done