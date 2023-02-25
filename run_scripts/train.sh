config=../configs/config_scene_graph.py
model_dir=/data/hulab/zcai75/checkpoints/pix2seq/scene_graph
export TF_CPP_MIN_LOG_LEVEL=3
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=1,2
# export NCCL_SOCKET_IFNAME=eno1,eth0
export AUTOGRAPH_VERBOSITY=0
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/data/hulab/zcai75/anaconda3/envs/tensorflow/

python ../run.py \
    --mode=train \
    --model_dir=$model_dir \
    --config=$config \
    --config.train.batch_size=48 \
    --config.optimization.learning_rate=3e-5 > train.out
