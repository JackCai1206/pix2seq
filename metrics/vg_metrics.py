from collections import defaultdict

import torch
from metrics.eval_utils.bounding_box import BoxList
from metrics.metric_registry import MetricRegistry
from metrics.eval_utils.vg_eval import do_vg_evaluation
import tensorflow as tf
from absl import logging
import json

@MetricRegistry.register('vg_sgg_recall')
class VGSGGRecallMetric():
    def __init__(self, config):
        self.config = config
        self.predictions = {'image_ids': [], 'box1': [], 'box2': [], 'box1_label': [], 'rel_label': [], 'box2_label': [], 'obj_scores': [], 'rel_scores': []}
        self.groundtruths = {'image_ids': [], 'box1': [], 'box2': [], 'box1_label': [], 'rel_label': [], 'box2_label': []}
        with open(self.config.dataset.vg_ann_label_file, 'r') as f:
            ann_label = json.load(f)
        print(ann_label.keys())
        self.ind_to_classes = ann_label['idx_to_label']
        self.ind_to_predicates = ann_label['idx_to_predicate']

    def record_prediction(self, **kwargs):
        for k, v in kwargs.items():
            self.predictions[k].extend(tf.squeeze(tf.unstack(v, axis=0)))

    def record_groundtruth(self, **kwargs):
        for k, v in kwargs.items():
            self.groundtruths[k].extend(tf.squeeze(tf.unstack(v, axis=0)))
    
    def result(self, step):
        predictions: dict[str, BoxList] = {}
        for i, img_id in enumerate(self.predictions['image_ids']):
            img_id = img_id.numpy()
            labels = tf.concat([self.predictions['box1_label'][i], self.predictions['box2_label'][i]], axis=0).numpy()
            boxes = tf.concat([self.predictions['box1'][i], self.predictions['box2'][i]], axis=0).numpy()
            obj_scores = tf.concat([self.predictions['obj_scores'][i][..., 0], self.predictions['obj_scores'][i][..., 1]], axis=0).numpy()
            rel_scores = self.predictions['rel_scores'][i].numpy()
            rel_tuples = tf.concat([self.predictions['box1_label'][i], self.predictions['box2_label'][i]], axis=0).numpy()

            # unique_idx = tf.unique(boxes)[1]
            # unique_boxes = tf.gather(boxes, unique_idx)
            predictions[img_id] = BoxList(torch.tensor(boxes), self.config.task.image_size, mode='xyxy')
            predictions[img_id] = predictions[img_id].resize(self.config.task.image_size)
            predictions[img_id].add_field('pred_labels', torch.tensor(labels))
            predictions[img_id].add_field('pred_scores', torch.tensor(obj_scores))
            predictions[img_id].add_field('rel_pair_idxs', torch.tensor(rel_tuples))
            predictions[img_id].add_field('pred_rel_scores', torch.tensor(rel_scores))

        groundtruths: dict[str, BoxList] = {}
        for i, img_id in enumerate(self.groundtruths['image_ids']):
            img_id = img_id.numpy()
            labels = tf.concat([self.groundtruths['box1_label'][i], self.groundtruths['box2_label'][i]], axis=0).numpy()
            boxes = tf.concat([self.groundtruths['box1'][i], self.groundtruths['box2'][i]], axis=0).numpy()
            rel_tuples = tf.concat([self.groundtruths['box1_label'][i], self.groundtruths['box2_label'][i]], axis=0).numpy()
            # unique_idx = tf.unique(labels)[1]
            # unique_boxes = tf.gather(boxes, unique_idx)
            groundtruths[img_id] = BoxList(torch.tensor(boxes), self.config.task.image_size, mode='xyxy')
            groundtruths[img_id] = groundtruths[img_id].resize(self.config.task.image_size)
            groundtruths[img_id].add_field('labels', torch.tensor(labels))
            groundtruths[img_id].add_field('gt_rels', torch.tensor(self.groundtruths['rel_label'][i].numpy()))
            groundtruths[img_id].add_field('relation_tuple', torch.tensor(rel_tuples))

        # tf.print(list(predictions.values())[0].bbox, list(groundtruths.values())[0].bbox)
        # tf.print(list(predictions.keys()), list(groundtruths.keys()))
        # tf.print(self.ind_to_classes, self.ind_to_predicates)
        mAP = do_vg_evaluation(self.ind_to_classes, self.ind_to_predicates, predictions, groundtruths, None, logging, ['bbox', 'relations'])

        return {'mAP': mAP}
    
    def reset_states(self):
        self.predictions = {'image_ids': [], 'box1': [], 'box2': [], 'box1_label': [], 'rel_label': [], 'box2_label': [], 'scores': []}
        self.groundtruths = {'image_ids': [], 'box1': [], 'box2': [], 'box1_label': [], 'rel_label': [], 'box2_label': []}
