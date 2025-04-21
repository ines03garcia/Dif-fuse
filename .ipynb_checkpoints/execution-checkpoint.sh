#!/bin/sh
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 # To avoid fragmentation

export CUDA_HOME=/opt/cuda-10.1.168_418_67/

export CUDNN_HOME=/opt/cuDNN-cuDNN-7.6.0.64_9.2/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

export NUM_GPUS=$2

echo $NUM_GPUS

cd . .
#export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:
source "/home/csantiago/venvs/diffuse_env/bin/activate"
#python scripts/image_train.py

# Batch size < 8, num_channels < 128 ( Batch size 2 channels 64 dá, batch size 4 channels 32 dá) 
# --diffusion_steps 100 deve ter mais a ver com o tempo de execução
# Falta ver com o "channel_mult"
python scripts/image_train.py --batch_size 2 --num_res_blocks 1 --num_channels 32 --channel_mult '1,2,2' --use_checkpoint 'True' --save_interval 1000



#python scripts/train_autoencoder.py -filepath_to_arguments_json_config scripts/baseline.json --experiment_name autoencoder

#python scripts/train_baseline_classifier.py -filepath_to_arguments_json_config scripts/baseline.json --experiment_name baseline_classifier

#python scripts/saliency_maps.py -filepath_to_arguments_json_config scripts/baseline.json --experiment_name create_saliency

#python scripts/image_sample_dif-fuse.py --model_path results/test/model054000.pt