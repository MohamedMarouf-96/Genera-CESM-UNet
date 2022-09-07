# rm -r cache_train
# rm -r cache_val
EXPNAME=cesm
# for ARCHNAME in unet2 unet3 unet4 unet5
for ARCHNAME in resnet18
do
    for NAME in low 
    do
        for EXPENAME in expA
        # for EXPENAME in expA expB expC
        do
            # rm -r cache_train
            # rm -r cache_val
            CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -n ${EXPNAME}_${ARCHNAME}_${EXPENAME}_${NAME} --unregistered --dataset_mode ${NAME} --batch-size 32 --gpu_ids 0,1,2,3 --epochs 10\
            --input_channels 1  --dataset cesm --experiment_type ${EXPENAME} --model_name ${ARCHNAME} --dataset_mode sub -l .0001
            # for EPOCH in 1 2 3 4 5
            # for EPOCH in 5 10
            # do
            #     CUDA_VISIBLE_DEVICES=0 python train.py -n  ${EXPNAME}_${ARCHNAME}_${EXPENAME}_${NAME} --unregistered --dataset_mode ${NAME} --load ./checkpoint/${EXPNAME}_${ARCHNAME}_${EXPENAME}_${NAME}/checkpoint_epoch${EPOCH}.pth \
            #     --input_channels 1 --dataset kaspernormA --experiment_type ${EXPENAME} --model_name ${ARCHNAME} --eval_only 
            #     CUDA_VISIBLE_DEVICES=0 python train.py -n  ${EXPNAME}_${ARCHNAME}_${EXPENAME}_${NAME} --unregistered --dataset_mode ${NAME} --load ./checkpoint/${EXPNAME}_${ARCHNAME}_${EXPENAME}_${NAME}/checkpoint_epoch${EPOCH}.pth \
            #     --input_channels 1 --dataset kaspernormA --experiment_type ${EXPENAME} --model_name ${ARCHNAME} --eval_only --train_set 
            # done
            # CUDA_VISIBLE_DEVICES=0 python train.py -n  ${EXPNAME}_${ARCHNAME}_${EXPENAME}_${NAME} --unregistered --dataset_mode ${NAME} --eval_only --load ./checkpoint/${EXPNAME}_${ARCHNAME}_${EXPENAME}_${NAME}/checkpoint_epoch5.pth \
            # --input_channels 1 --dataset kaspernormA --experiment_type ${EXPENAME} --full_set --model_name ${ARCHNAME}
            # rm -r cache_train
            # rm -r cache_val
        done
    done
done