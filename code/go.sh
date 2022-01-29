

CUDA_LAUNCH_BLOCKING=1 python train.py --final_epochs=10 --crossval_size=250
CUDA_LAUNCH_BLOCKING=1 python train.py --final_epochs=10 --test --crossval_size=250 --crossval_load --load_model
