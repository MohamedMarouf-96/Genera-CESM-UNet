# rm -r cache_train
# rm -r cache_val
EXPNAME=segarch
# for ARCHNAME in unet2 unet3 unet4 unet5
for ARCHNAME in unet
do
    for NAME in A
    do
        for EXPENAME in expA
        # for EXPENAME in expA expB expC
        do
            # rm -r cache_train
            # rm -r cache_val
            CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n ${EXPNAME}_${ARCHNAME}_${EXPENAME}_${NAME} --unregistered --dataset_mode ${NAME} --batch-size 32 --gpu_ids 0,1,2,3 --epochs 5 \
            --input_channels 1  --dataset kaspernormA --experiment_type ${EXPENAME} --balanced --model_name ${ARCHNAME}
            CUDA_VISIBLE_DEVICES=0 python train.py -n  ${EXPNAME}_${ARCHNAME}_${EXPENAME}_${NAME} --unregistered --dataset_mode ${NAME} --eval_only --load ./checkpoint/${EXPNAME}_${ARCHNAME}_${EXPENAME}_${NAME}/checkpoint_epoch5.pth \
            --input_channels 1 --dataset kaspernormA --experiment_type ${EXPENAME} --model_name ${ARCHNAME}
            CUDA_VISIBLE_DEVICES=0 python train.py -n  ${EXPNAME}_${ARCHNAME}_${EXPENAME}_${NAME} --unregistered --dataset_mode ${NAME} --eval_only --load ./checkpoint/${EXPNAME}_${ARCHNAME}_${EXPENAME}_${NAME}/checkpoint_epoch5.pth \
            --input_channels 1 --dataset kaspernormA --experiment_type ${EXPENAME} --full_set --model_name ${ARCHNAME}
            # rm -r cache_train
            # rm -r cache_val
        done
    done
done