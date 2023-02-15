import code
import collections
import json
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
from tqdm import tqdm
import vocab
from data.scripts import tfrecord_lib
import tensorflow as tf
import h5py

flags.DEFINE_string('vg_image_dir', '', 'Directory containing images.')
flags.DEFINE_string('vg_ann_file', '', 'Instance annotation file.')
flags.DEFINE_string('image_data_file', '', 'Image data file.')
flags.DEFINE_string('vg_ann_label_file', '', 'Json containing label information')
flags.DEFINE_bool('train', False, 'Is train split.')
flags.DEFINE_string('output_dir', '', 'Output directory')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')
FLAGS = flags.FLAGS

def create_anno_iter(img_data, label_data, ann):
	skipped = 0
	for j, img in enumerate(tqdm(img_data)):
		split = 0 if FLAGS.train else 2
		if ann['split'][j] != split:
			skipped += 1
			continue

		with open(os.path.join(FLAGS.vg_image_dir, str(img['image_id'])) + '.jpg', 'rb') as fid:
			encoded_jpg = fid.read()
		feature_dict = tfrecord_lib.image_info_to_feature_dict(
			img['height'], img['width'], f"{img['image_id']}.JPG", img['image_id'], encoded_jpg, 'jpg'
		)

		first_rel = ann['img_to_first_rel'][j]
		last_rel = ann['img_to_last_rel'][j]
		img_rels = ann['relationships'][first_rel : last_rel+1]
		if len(img_rels) == 0:
			skipped += 1
			continue

		box1_ids = img_rels[:, 0]
		box2_ids = img_rels[:, 1]
		pred_ids = ann['predicates'][first_rel : last_rel+1]
		box1 = tf.constant([ann['boxes_1024'][i] for i in box1_ids])
		box2 = tf.constant([ann['boxes_1024'][i] for i in box2_ids])
		pred_label = [label_data['idx_to_predicate'][str(i[0])].encode('utf-8') for i in pred_ids]
		box1_label = [label_data['idx_to_label'][str(ann['labels'][i][0])].encode('utf-8') for i in box1_ids]
		box2_label = [label_data['idx_to_label'][str(ann['labels'][i][0])].encode('utf-8') for i in box2_ids]

		feature_dict.update({
			'box1': tfrecord_lib.convert_to_feature(tf.io.serialize_tensor(box1).numpy(), 'bytes'),
			'pred': tfrecord_lib.convert_to_feature(pred_ids, 'int64_list'),
			'box2': tfrecord_lib.convert_to_feature(tf.io.serialize_tensor(box2).numpy(), 'bytes'),
			'box1_id': tfrecord_lib.convert_to_feature(box1_ids, 'int64_list'),
			'box2_id': tfrecord_lib.convert_to_feature(box2_ids, 'int64_list'),
			'pred_label': tfrecord_lib.convert_to_feature(pred_label, 'bytes_list'),
			'box1_label': tfrecord_lib.convert_to_feature(box1_label, 'bytes_list'),
			'box2_label': tfrecord_lib.convert_to_feature(box2_label, 'bytes_list')
		})

		yield feature_dict, skipped
		skipped = 0
		# break

def create_example(feature_dict, skipped):
	example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
	return example, skipped

def main(_):
	logging.info('Building instance index.')

	directory = os.path.dirname(FLAGS.output_dir)
	if not os.path.isdir(directory):
		os.makedirs(directory, exist_ok=True)

	ann = h5py.File(FLAGS.vg_ann_file, 'r')
	with open(FLAGS.image_data_file, 'r') as img_data_f, open(FLAGS.vg_ann_label_file, 'r') as label_f:
		img_data = json.load(img_data_f)
		label_data = json.load(label_f)
		print(ann.keys())
		print(img_data[0].keys())
		print(label_data.keys())

	anno_iter = create_anno_iter(img_data, label_data, ann)

	tfrecord_lib.write_tf_record_dataset(
		output_path=FLAGS.output_dir,
		annotation_iterator=anno_iter,
		process_func=create_example,
		num_shards=FLAGS.num_shards,
	multiple_processes=8)

def run_main():
	app.run(main)

if __name__ == '__main__':
  	run_main()
