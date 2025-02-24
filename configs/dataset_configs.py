# coding=utf-8
# Copyright 2022 The Pix2Seq Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Dataset configs."""
import os
from configs.config_base import D


_shared_dataset_config = D(
    batch_duplicates=1,
    cache_dataset=True,
)

# Generate tfrecords for the dataset using data/scripts/create_coco_tfrecord.py
# and add paths here.
COCO_TRAIN_TFRECORD_PATTERN = 'gs://pix2seq/multi_task/data/coco/tfrecord/train*'
COCO_VAL_TFRECORD_PATTERN = '/data/hulab/zcai75/pix2seq/coco/val*'

VG_TRAIN_TFRECORD_PATTERN = '/data/hulab/zcai75/pix2seq/vg/train*'
VG_VAL_TFRECORD_PATTERN = '/data/hulab/zcai75/pix2seq/vg/val*'

# Download from gs://pix2seq/multi_task/data/coco/json
COCO_ANNOTATIONS_DIR = '/data/hulab/zcai75/coco/annotations/'

_shared_coco_dataset_config = D(
    train_file_pattern=COCO_TRAIN_TFRECORD_PATTERN,
    val_file_pattern=COCO_VAL_TFRECORD_PATTERN,
    train_num_examples=118287,
    eval_num_examples=5000,
    train_split='train',
    eval_split='validation',
    # Directory of annotations used by the metrics.
    # Also need to set train_filename_for_metrics and val_filename_for_metrics.
    # If unset, groundtruth annotations should be specified via
    # record_groundtruth.
    coco_annotations_dir_for_metrics=COCO_ANNOTATIONS_DIR,
    label_shift=0,
    **_shared_dataset_config
)

vg_dataset_config = D(
    train_file_pattern=VG_TRAIN_TFRECORD_PATTERN,
    val_file_pattern=VG_VAL_TFRECORD_PATTERN,
    # train_num_examples=108073-12928,
    # eval_num_examples=108073-81627,
    train_num_examples=1000,
    eval_num_examples=14,
    train_split='train',
    eval_split='validation',
    **_shared_dataset_config,
    num_obj_classes=151,
    num_rel_classes=51,
)

dataset_configs = {
    'coco/2017_object_detection':
        D(
            name='coco/2017_object_detection',
            train_filename_for_metrics='instances_train2017.json',
            val_filename_for_metrics='instances_val2017.json',
            category_names_path=os.path.join(
                _shared_coco_dataset_config['coco_annotations_dir_for_metrics'],
                'instances_val2017.json'),
            **_shared_coco_dataset_config
        ),
    'coco/2017_instance_segmentation':
        D(
            name='coco/2017_instance_segmentation',
            train_filename_for_metrics='instances_train2017.json',
            val_filename_for_metrics='instances_val2017.json',
            category_names_path=os.path.join(
                _shared_coco_dataset_config['coco_annotations_dir_for_metrics'],
                'instances_val2017.json'),
            **_shared_coco_dataset_config
        ),
    'coco/2017_keypoint_detection':
        D(
            name='coco/2017_keypoint_detection',
            train_filename_for_metrics='person_keypoints_train2017.json',
            val_filename_for_metrics='person_keypoints_val2017.json',
            category_names_path=os.path.join(
                _shared_coco_dataset_config['coco_annotations_dir_for_metrics'],
                'person_keypoints_val2017.json'),
            **_shared_coco_dataset_config
        ),
    'coco/2017_captioning':
        D(name='coco/2017_captioning',
          train_filename_for_metrics='captions_train2017_eval_compatible.json',
          val_filename_for_metrics='captions_val2017_eval_compatible.json',
          **_shared_coco_dataset_config),

    'visual_genome':
        D(
            name='visual_genome',
            vg_image_dir='/data/hulab/zcai75/visual_genome/VG_100K',
            vg_ann_file='/data/hulab/zcai75/visual_genome/VG-SGG-with-attri.h5',
            image_data_file='/data/hulab/zcai75/visual_genome/image_data.json',
            vg_ann_label_file='/data/hulab/zcai75/visual_genome/vg_motif_anno/VG-SGG-dicts-with-attri.json',
            **vg_dataset_config
        ),
}
