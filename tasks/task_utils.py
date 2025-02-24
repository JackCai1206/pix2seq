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
"""Common task utils."""

import json
from typing import Optional, Any, Dict
from absl import logging
import utils
import vocab
import tensorflow as tf


def get_category_names(
    category_names_path: Optional[str]) -> Dict[int, Dict[str, Any]]:
  """Returns dictionary of category names.

  Args:
    category_names_path: Path to category names json. Expected to be a json file
      with the format {"categories": [{"id": 1, "name": "Person"}, ...]}. If not
      specified, the category id is used as the name.

  Returns:
    Dictionary with the format {1: {"name": "Person"}, ...}
  """
  if not category_names_path:
    logging.info(
        'category_names_path not specified, default category names will be used'
    )
    return {i: {'name': str(i)} for i in range(10000)}
  logging.info('Loading category names from %s', category_names_path)
  category_names = {}
  with tf.io.gfile.GFile(category_names_path, 'r') as f:
    annotations = json.load(f)
  category_names = {c['id']: c for c in annotations['categories']}
  return category_names

def get_category_names_from_vg_label_file(path):
  with open(path, 'r') as f:
    ann = json.load(f)
  return ann['idx_to_label']

def build_instance_prompt_seq(task_vocab_id: int, bbox, label,
                              quantization_bins, coord_vocab_shift):
  """"Build prompt seq for instance tasks like instance segmentation, keypoints.

  Args:
    task_vocab_id: Vocab id for the task.
    bbox: `float` bounding box of shape (bsz, n, 4).
    label: `int` label of shape (bsz, n).
    quantization_bins: `int`.
    coord_vocab_shift: `int`, shifting coordinates by a specified integer.

  Returns:
    discrete prompt sequence of (task_id, bbox, label) with shape (bsz, n, 6).
    tokens are zero'ed if label is padding (0).
  """
  task_id = tf.constant(task_vocab_id)
  quantized_bbox = utils.quantize(bbox, quantization_bins)
  quantized_bbox = quantized_bbox + coord_vocab_shift
  new_label = tf.expand_dims(label + vocab.BASE_VOCAB_SHIFT, -1)
  prompt_seq = tf.concat([quantized_bbox, new_label], axis=-1)
  task_id = tf.zeros_like(prompt_seq[..., :1]) + tf.cast(task_id, label.dtype)
  prompt_seq = tf.concat([task_id, prompt_seq], -1)
  is_padding = tf.expand_dims(tf.equal(label, 0), -1)
  prompt_seq = tf.where(is_padding, tf.zeros_like(prompt_seq), prompt_seq)
  return prompt_seq


def build_instance_response_seq_from_points(points, label, quantization_bins,
                                            coord_vocab_shift):
  """"Build target seq for instance tasks like instance segmentation, keypoints.

  Args:
    points: `float` points of shape (bsz, n, k).
    label: `int` label of shape (bsz, n).
    quantization_bins: `int`.
    coord_vocab_shift: `int`, shifting coordinates by a specified integer.

  Returns:
    discrete target sequence with shape (bsz, n, k). tokens are zero'ed
    if label is padding (0).
  """
  quantized_points = utils.quantize(points, quantization_bins)
  quantized_points = quantized_points + coord_vocab_shift
  response_seq = utils.replace_reserved_tokens(
      quantized_points, points, vocab.FLOAT_TO_TOKEN)
  is_padding = tf.expand_dims(tf.equal(label, 0), -1)
  response_seq = tf.where(is_padding, tf.zeros_like(response_seq), response_seq)
  return response_seq


def build_prompt_seq_from_task_id(task_vocab_id: int,
                                  response_seq=None,
                                  prompt_shape=None):
  """"Build prompt seq just using task id.

  Args:
    task_vocab_id: Vocab id for the task.
    response_seq: an (optional) discerte target sequen with shape (bsz, ..., k).
    prompt_shape: an (optional) tuple for prompt shape. One and only one of
      `response_seq` and `prompt_shape` should be specified.

  Returns:
    discrete input sequence of task id with shape (bsz, ..., 1).
  """
  task_id = tf.constant(task_vocab_id)
  if response_seq is not None:
    prompt_seq = tf.zeros_like(response_seq[..., :1]) + tf.cast(
        task_id, response_seq.dtype)
  if prompt_shape is not None:
    assert response_seq is None, 'double specification'
    prompt_seq = tf.zeros(prompt_shape, dtype=tf.int64) + tf.cast(
        task_id, dtype=tf.int64)
  return prompt_seq


def decode_instance_seq_to_points(seq, quantization_bins, coord_vocab_shift):
  """Decode points for seq from `build_instance_response_seq_from_points`."""
  assert seq.dtype in (tf.int64, tf.int32)
  points = seq - coord_vocab_shift
  points = utils.dequantize(points, quantization_bins)
  return utils.replace_reserved_tokens(points, seq, vocab.TOKEN_TO_FLOAT)


def decode_object_seq_to_bbox(logits,
                              pred_seq,
                              quantization_bins,
                              coord_vocab_shift):
  """Decode objects (label & bbox) for seq from `build_response_seq_from_bbox`.

  Assume yxyxc format with truncation at the end for any uneven extra tokens.
    Replace class tokens with argmax instead of sampling.

  Args:
    logits: `float` output logits in shape of (bsz, max_seq_len, vocab_size).
    pred_seq: `int` pred sequence in shape of (bsz, max_seq_len).
    quantization_bins: `int` for bins.
    coord_vocab_shift: `int`, shifting coordinates by a specified integer.

  Returns:
    pred_class: `int` of shape (bsz, max_instances_per_image).
    pred_bbox: `float` of shape (bsz, max_instances_per_image, 4).
    pred_score: `float` of shape (bsz, max_instances_per_image).
  """
  _, seqlen, vocab_size = logits.shape
  if seqlen % 5 != 0:  # truncate out the last few tokens.
    pred_seq = pred_seq[..., :-(seqlen % 5)]
    logits = logits[..., :-(seqlen % 5), :]
  pred_class_p = tf.nn.softmax(logits)[:, 4::5]  # (bsz, instances, vocab_size)
  mask_s1 = [0.] * vocab.BASE_VOCAB_SHIFT  # reserved.
  mask_s2 = [1.] * (coord_vocab_shift - vocab.BASE_VOCAB_SHIFT)  # labels.
  mask_s3 = [0] * (vocab_size - coord_vocab_shift)  # coordinates and others.
  mask = tf.constant(mask_s1 + mask_s2 + mask_s3)
  pred_class = tf.argmax(pred_class_p * mask[tf.newaxis, tf.newaxis, :], -1)
  pred_score = tf.reduce_sum(
      pred_class_p * tf.one_hot(pred_class, vocab_size), -1)
  pred_class = tf.maximum(pred_class - vocab.BASE_VOCAB_SHIFT, 0)
  pred_bbox = seq_to_bbox(pred_seq - coord_vocab_shift, quantization_bins)
  return pred_class, pred_bbox, pred_score


def decode_seq_to_triplets(logits,
                              pred_seq,
                              quantization_bins,
                              coord_vocab_shift,
                              num_obj_classes, num_rel_classes):
  _, seqlen, vocab_size = logits.shape
  triplet_len = 15
  print(pred_seq[0,:15], pred_seq.shape)
  if seqlen % triplet_len != 0:  # truncate out the last few tokens.
    pred_seq = pred_seq[..., :-(seqlen % triplet_len)]
    logits = logits[..., :-(seqlen % triplet_len), :]
  # Chunk the sequence into triplets (length 12).
  pred_seq = tf.reshape(pred_seq, [-1, pred_seq.shape[-1] // triplet_len, triplet_len])
  logits = tf.reshape(logits, [-1, logits.shape[-2] // triplet_len, triplet_len, logits.shape[-1]])
  # print(pred_seq.shape)
  pred_obj_class_logits = tf.gather(logits, indices=[4, 12], axis=2)
  pred_rel_class_logits = tf.gather(logits, indices=[6], axis=2)
  pred_obj_class_p = tf.nn.softmax(pred_obj_class_logits) # (bsz, instances, triplet_len, vocab_size)
  pred_rel_class_p = tf.nn.softmax(pred_rel_class_logits)
  # print(pred_obj_class_p.shape)
  # print(pred_rel_class_p.shape)
  mask_s1 = [0.] * vocab.BASE_VOCAB_SHIFT  # reserved.
  mask_s_rel = [0.] * num_obj_classes + [1.] * num_rel_classes + [0.] * (coord_vocab_shift - vocab.BASE_VOCAB_SHIFT - num_obj_classes - num_rel_classes)  # labels.
  mask_s_obj = [1.] * num_obj_classes + [0.] * num_rel_classes + [0.] * (coord_vocab_shift - vocab.BASE_VOCAB_SHIFT - num_obj_classes - num_rel_classes)
  mask_s3 = [0] * (vocab_size - coord_vocab_shift)  # coordinates and others.
  mask_obj = tf.constant(mask_s1 + mask_s_obj + mask_s3)
  mask_rel = tf.constant(mask_s1 + mask_s_rel + mask_s3)
  # print(mask.shape)
  pred_obj_class = tf.argmax(pred_obj_class_p * mask_obj[tf.newaxis, tf.newaxis, tf.newaxis, :], -1)
  pred_rel_class = tf.argmax(pred_rel_class_p * mask_rel[tf.newaxis, tf.newaxis, tf.newaxis, :], -1)
  # print(pred_class.shape)
  pred_obj_score = tf.reduce_sum(
      pred_obj_class_p * tf.one_hot(pred_obj_class, vocab_size), -1)
  # pred_rel_score = tf.reduce_sum(
  #     pred_rel_class_p * tf.one_hot(pred_rel_class, vocab_size), -1)a
  pred_rel_score = pred_rel_class_p
  # print(pred_obj_score.shape)
  # print(pred_rel_score.shape)
  pred_obj_class = tf.maximum(pred_obj_class - vocab.BASE_VOCAB_SHIFT, 0)
  pred_rel_class = tf.maximum(pred_rel_class - vocab.BASE_VOCAB_SHIFT - num_obj_classes, 0)
  # print(pred_class.shape)
  pred_bbox1, pred_bbox2 = triplet_seq_to_bbox(pred_seq - coord_vocab_shift, quantization_bins)
  # tf.print(pred_bbox1, pred_bbox2)

  box1_class, box2_class = tf.split(pred_obj_class, 2, axis=-1)
  rel_class = pred_rel_class

  return box1_class, rel_class, box2_class, pred_bbox1, pred_bbox2, pred_obj_score, pred_rel_score

def seq_to_bbox(seq, quantization_bins, seq_format='yxyx_name'):
  """Returns [0, 1] normalized yxyx bbox from token sequence."""
  # [batch, 5*num_instances]
  assert seq.shape.rank == 2, seq.shape.as_list()
  # [batch, num_instances, 1]
  if seq_format.startswith('name'):
    ymin = tf.expand_dims(seq[:, 1::5], -1)
    xmin = tf.expand_dims(seq[:, 2::5], -1)
    ymax = tf.expand_dims(seq[:, 3::5], -1)
    xmax = tf.expand_dims(seq[:, 4::5], -1)
  else:
    ymin = tf.expand_dims(seq[:, 0::5], -1)
    xmin = tf.expand_dims(seq[:, 1::5], -1)
    ymax = tf.expand_dims(seq[:, 2::5], -1)
    xmax = tf.expand_dims(seq[:, 3::5], -1)
  if seq_format in ['name_cycxhw', 'cycxhw_name']:
    ycnt, xcnt, ysize, xsize = ymin, xmin, ymax, xmax
    ymin = ycnt - ysize//2
    xmin = xcnt - xsize//2
    ymax = ycnt + ysize//2
    xmax = xcnt + xsize//2
  quantized_box = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
  quantized_box = utils.dequantize(quantized_box, quantization_bins)
  return tf.minimum(tf.maximum(quantized_box, 0), 1)

def triplet_seq_to_bbox(seq, quantization_bins, seq_format='yxyx_name'):
  """Returns [0, 1] normalized yxyx bbox from token sequence."""
  # [batch, num_instances, triplet_len]
  box1_seq = tf.reshape(tf.gather(seq, indices=[0, 1, 2, 3, 4], axis=2), [-1, 5 * seq.shape[1]])
  box2_seq = tf.reshape(tf.gather(seq, indices=[8, 9, 10, 11, 12], axis=2), [-1, 5 * seq.shape[1]])
  # print(seq[0,0])
  # print(box1_seq.shape)

  box1 = seq_to_bbox(box1_seq, quantization_bins, seq_format)
  box2 = seq_to_bbox(box2_seq, quantization_bins, seq_format)
  # print(box1.shape)
  return box1, box2

def compute_weighted_scores(bbox_scores, pred_seq, logits,
                            points_score_weight):
  """Computes per instance score as weighted sum of box score and mean pred_seq score."""
  probs = tf.nn.softmax(logits, axis=-1)
  # Set 0 weight for padding tokens.
  token_weight = tf.where(tf.equal(pred_seq, vocab.PADDING_TOKEN), 0.0, 1.0)
  likelihoods = tf.gather(probs, pred_seq, batch_dims=pred_seq.shape.rank)
  points_score = (
      tf.reduce_sum(likelihoods * token_weight, axis=-1) /
      tf.reduce_sum(token_weight, axis=-1))
  num_instances_in_batch = bbox_scores.shape[0]
  num_samples = points_score.shape[0] // num_instances_in_batch
  points_score = tf.reshape(points_score, [num_instances_in_batch, num_samples])
  points_score = tf.reduce_mean(points_score, axis=-1)
  return (points_score_weight * points_score +
          (1 - points_score_weight) * bbox_scores)


def join_if_not_none(args, sep):
  args = [str(arg) for arg in args if arg is not None]
  return sep.join(args)

