EXPNAME=KaspernormA
for NAME in A B
do
    # for EXPENAME in expB
    for EXPENAME in expA expB expC
    do
        CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n ${EXPNAME}_${EXPENAME}_${NAME} --unregistered --dataset_mode ${NAME} --batch-size 32 --gpu_ids 0,1,2,3 --epochs 5 \
        --input_channels 1  --dataset kaspernormA --experiment_type ${EXPENAME}
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n ${EXPNAME}_${NAME} --unregistered --dataset_mode ${NAME} --batch-size 32 --gpu_ids 0,1,2,3 --epochs 5 \
# --input_channels 1  --dataset kaspernormA --eval_only --load ./checkpoint/full2d_balanced_${NAME}/checkpoint_epoch5.pth
        # CUDA_VISIBLE_DEVICES=0 python train.py -n  ${EXPNAME}_${EXPENAME}_${NAME} --unregistered --dataset_mode ${NAME} --eval_only --load ./checkpoint/full2d_balanced_${NAME}/checkpoint_epoch5.pth \
        # --input_channels 1 --dataset kaspernormA --experiment_type ${EXPENAME}
        # CUDA_VISIBLE_DEVICES=0 python train.py -n  ${EXPNAME}_${EXPENAME}_${NAME} --unregistered --dataset_mode ${NAME} --eval_only --load ./checkpoint/${EXPNAME}_${EXPENAME}_${NAME}/checkpoint_epoch5.pth \
        # --input_channels 1 --dataset kaspernormA --experiment_type ${EXPENAME}
    done
done