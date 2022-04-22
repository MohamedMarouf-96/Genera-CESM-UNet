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
for NAME in A
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n dn${NAME} --unregistered --dataset_mode dn${NAME} --batch-size 16 --gpu_ids 0,1,2,3 --epochs 5
    CUDA_VISIBLE_DEVICES=0 python train.py -n dn${NAME} --unregistered --dataset_mode dn${NAME} --eval_only --load ./checkpoint/dn${NAME}/checkpoint_epoch5.pth
done
# for NAME in A
# do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n dn${NAME}_blocking --unregistered --dataset_mode dn${NAME} --batch-size 16 --gpu_ids 0,1,2,3 --epochs 5
#     CUDA_VISIBLE_DEVICES=0 python train.py -n dn${NAME}_blocking --unregistered --dataset_mode dn${NAME} --eval_only --load ./checkpoint/dn${NAME}_blocking/checkpoint_epoch5.pth
# done