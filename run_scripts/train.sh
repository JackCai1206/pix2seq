config=../configs/config_scene_graph.py
model_dir=/data/hulab/zcai75/checkpoints/pix2seq/scene_graph

python ../run.py \
    --mode=train \
    --model_dir=$model_dir \
    --config=$config \
    --config.train.batch_size=32 \
    --config.train.epochs=20 \
    --config.optimization.learning_rate=3e-5 \
    --run_eagerly > train.out
