# if registered
# NAME=B
# CUDA_VISIBLE_DEVICES=0 python train.py -n $NAME --dataset_mode $NAME --eval_only --load ./checkpoint/$NAME/checkpoint_epoch5.pth

# if unregistered
# NAME=A
# CUDA_VISIBLE_DEVICES=0 python train.py -n ${NAME}_un --dataset_mode $NAME --unregistered --eval_only --load ./checkpoint/${NAME}_un/checkpoint_epoch5.pth


# for NAME in A ABD
for NAME in ABD
do
    CUDA_VISIBLE_DEVICES=0 python train.py -n $NAME --dataset_mode $NAME --eval_only --load ./checkpoint/$NAME/checkpoint_epoch5.pth
    CUDA_VISIBLE_DEVICES=0 python train.py -n $NAME --dataset_mode $NAME --eval_only --unregistered --load ./checkpoint/$NAME/checkpoint_epoch5.pth
    CUDA_VISIBLE_DEVICES=0 python train.py -n ${NAME}_un --dataset_mode $NAME --eval_only --load ./checkpoint/${NAME}_un/checkpoint_epoch5.pth
    CUDA_VISIBLE_DEVICES=0 python train.py -n ${NAME}_un --dataset_mode $NAME --unregistered --eval_only --load ./checkpoint/${NAME}_un/checkpoint_epoch5.pth
done

# NAME=B
# CUDA_VISIBLE_DEVICES=0 python train.py -n $NAME --dataset_mode $NAME --eval_only --load ./checkpoint/$NAME/checkpoint_epoch5.pth
# NAME=AA
# CUDA_VISIBLE_DEVICES=0 python train.py -n $NAME --dataset_mode $NAME --eval_only --load ./checkpoint/$NAME/checkpoint_epoch5.pth
