export PYTHONPATH=$PYTHONPATH:~/Github/pix2seq
train=true
if $train; then
    split=train
else
    split=val
fi

python ../data/scripts/create_vg_tfrecord.py \
    --vg_image_dir /data/hulab/zcai75/visual_genome/VG_100K \
    --vg_ann_file /data/hulab/zcai75/visual_genome/VG-SGG-with-attri.h5 \
    --image_data_file /data/hulab/zcai75/visual_genome/image_data.json \
    --vg_ann_label_file /data/hulab/zcai75/visual_genome/vg_motif_anno/VG-SGG-dicts-with-attri.json \
    --train=${train} \
    --output_dir /data/hulab/zcai75/pix2seq/vg/${split} \
    --num_shards=1
