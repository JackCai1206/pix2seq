export PYTHONPATH=$PYTHONPATH:~/Github/pix2seq
split=val

python ../data/scripts/create_coco_tfrecord.py \
    --coco_image_dir /data/hulab/zcai75/coco/images/${split}2017 \
    --ins_ann_file /data/hulab/zcai75/coco/annotations/instances_${split}2017.json \
    --cap_ann_file /data/hulab/zcai75/coco/annotations/captions_${split}2017.json \
    --key_ann_file /data/hulab/zcai75/coco/annotations/person_keypoints_${split}2017.json \
    --pan_ann_file /data/hulab/zcai75/coco/annotations/panoptic_${split}2017.json \
    --pan_masks_dir /data/hulab/zcai75/coco/annotations/panoptic_${split}2017 \
    --output_dir /data/hulab/zcai75/pix2seq/coco/${split}
