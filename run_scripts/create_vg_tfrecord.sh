export PYTHONPATH=$PYTHONPATH:~/Github/pix2seq
split=$1

# if split is not train or val, then exit
if [ "$split" != "train" ] && [ "$split" != "val" ]; then
    echo "split must be train or val"
    exit
fi

python ../data/scripts/create_vg_tfrecord.py \
    --vg_image_dir /data/hulab/zcai75/visual_genome/VG_100K \
    --vg_ann_file /data/hulab/zcai75/visual_genome/VG-SGG-with-attri.h5 \
    --image_data_file /data/hulab/zcai75/visual_genome/image_data.json \
    --vg_ann_label_file /data/hulab/zcai75/visual_genome/vg_motif_anno/VG-SGG-dicts-with-attri.json \
    --split=${split} \
    --output_dir /data/hulab/zcai75/pix2seq/vg/${split} \
    --num_shards=32
