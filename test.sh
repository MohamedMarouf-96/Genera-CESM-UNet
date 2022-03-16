# if registered
# NAME=B
# CUDA_VISIBLE_DEVICES=0 python train.py -n $NAME --dataset_mode $NAME --eval_only --load ./checkpoint/$NAME/checkpoint_epoch5.pth

# if unregistered
# NAME=A
# CUDA_VISIBLE_DEVICES=0 python train.py -n ${NAME}_un --dataset_mode $NAME --unregistered --eval_only --load ./checkpoint/${NAME}_un/checkpoint_epoch5.pth


# for NAME in A ABD
# for NAME in ABD
# # for NAME in A
# do
#     # CUDA_VISIBLE_DEVICES=0 python train.py -n $NAME --dataset_mode $NAME --eval_only --load ./checkpoint/$NAME/checkpoint_epoch5.pth
#     # CUDA_VISIBLE_DEVICES=0 python train.py -n $NAME --dataset_mode $NAME --eval_only --unregistered --load ./checkpoint/$NAME/checkpoint_epoch5.pth
#     # CUDA_VISIBLE_DEVICES=0 python train.py -n ${NAME}_un --dataset_mode $NAME --eval_only --load ./checkpoint/${NAME}_un/checkpoint_epoch5.pth
#     CUDA_VISIBLE_DEVICES=0 python train.py -n ${NAME}_un --dataset_mode $NAME --unregistered --eval_only --load ./checkpoint/${NAME}_un/checkpoint_epoch5.pth
# done
# CUDA_VISIBLE_DEVICES=0 python train.py -n A_with_conf --dataset_mode A --eval_only --load ./checkpoint/A_with_conf/checkpoint_epoch1.pth
    # CUDA_VISIBLE_DEVICES=0 python train.py -n A_with_conf_freeze_un --unregistered --dataset_mode A --eval_only --load ./checkpoint/A_with_conf_freeze_un/checkpoint_epoch1.pth
# NAME=B
# CUDA_VISIBLE_DEVICES=1 python train.py -n $NAME --dataset_mode B --eval_only --load ./checkpoint/$NAME/checkpoint_epoch5.pth
# NAME=AA
# CUDA_VISIBLE_DEVICES=0 python train.py -n $NAME --dataset_mode $NAME --eval_only --load ./checkpoint/$NAME/checkpoint_epoch5.pth
for NAME in A B
do
    CUDA_VISIBLE_DEVICES=0 python train.py -n ${NAME}_with_conf_freeze --dataset_mode ${NAME} --eval_only --load ./checkpoint/${NAME}_with_conf_freeze/checkpoint_epoch1.pth
done
