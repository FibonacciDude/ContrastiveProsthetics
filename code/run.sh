
CUDA_LAUNCH_BLOCKING=1 python load.py --load --no_glove
python train.py --prediction --crossval_size=100 --batch_size=16 --final_epochs=15
