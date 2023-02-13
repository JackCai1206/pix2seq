config=../configs/config_scene_graph.py
model_dir=/data/hulab/zcai75/checkpoints/pix2seq/scene_graph
export TF_CPP_MIN_LOG_LEVEL=1
# export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=1
# export AUTOGRAPH_VERBOSITY=0

python ../run.py \
    --mode=train \
    --model_dir=$model_dir \
    --config=$config \
    --config.train.batch_size=12 \
    --config.train.epochs=2 \
    --config.optimization.learning_rate=3e-5 \
    --run_eagerly > train.out
