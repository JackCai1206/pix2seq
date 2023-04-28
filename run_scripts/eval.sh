config=../configs/config_scene_graph.py
model_dir=/data/hulab/zcai75/checkpoints/pix2seq/scene_graph
# Path to save the detected boxes for evaluating other tasks.
export TF_CPP_MIN_LOG_LEVEL=2
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=2
# export AUTOGRAPH_VERBOSITY=0

boxes_json_path=$model_dir/boxes.json
python ../run.py \
    --config=$config \
    --model_dir=$model_dir \
    --mode=eval \
    --run_eagerly=True