python ../run.py \
    --model_dir /data/hulab/zcai75/checkpoints/pix2seq/coco_det_finetune \
    --mode eval

config=configs/config_multi_task.py:object_detection@coco/2017_object_detection,vit-b
model_dir=/tmp/pix2seq_eval_det
# Path to save the detected boxes for evaluating other tasks.
boxes_json_path=$model_dir/boxes.json
python3 run.py --config=$config --model_dir=$model_dir --mode=eval --config.task.eval_outputs_json_path=$boxes_json_path