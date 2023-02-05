import numpy as np
import utils
import vocab
from data import dataset as dataset_lib
from data import decode_utils
import tensorflow as tf

@dataset_lib.DatasetRegistry.register('visual_genome')
class VisualGenomeTFRecordDataset(dataset_lib.TFRecordDataset):
    def get_feature_map(self):
        image_feature_map = decode_utils.get_feature_map_for_image()
        vg_feature_map = {
			'box1': tf.io.VarLenFeature(tf.float32),
			'pred': tf.io.VarLenFeature(tf.int64),
			'box2': tf.io.VarLenFeature(tf.float32),
			'pred_label': tf.io.VarLenFeature(tf.int64),
			'box1_label': tf.io.VarLenFeature(tf.int64),
			'box2_label': tf.io.VarLenFeature(tf.int64)
		}
    
    def filter_example(self, example, training):
        if training:
            return tf.shape(example['box1'])[0] > 0
        else:
            return True
    
    def extract(self, example, training):
        features = {
            'image': decode_utils.decode_image(example),
            'image/id': tf.strings.to_number(example['image/source_id'], tf.int64),
        }

        scale = 1. / utils.tf_float32(tf.shape(features['image'])[:2])
        box1 = utils.scale_points(example['box1'])
        box2 = utils.scale_points(example['box2'])

        labels = {k: v for k, v in example.items() if 'image' not in k}

        return features, labels
