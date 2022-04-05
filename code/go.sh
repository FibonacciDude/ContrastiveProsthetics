
# 0.3485771417617798
#CUDA_LAUNCH_BLOCKING=1 python train.py --final_epochs=20 --crossval_size=150 --batch_size=8 --crossval_load  --test
# 0.36178848147392273 
# CUDA_LAUNCH_BLOCKING=1 python train.py --final_epochs=20 --crossval_size=150 --batch_size=8 --crossval_load  --test --no_adabn
CUDA_LAUNCH_BLOCKING=1 python train.py --final_epochs=8 --crossval_size=150 --batch_size=8 --crossval_load  --test --no_adabn
#CUDA_LAUNCH_BLOCKING=1 python train.py --final_epochs=10 --test --crossval_size=250 --batch_size=8 --crossval_load #--load_model
