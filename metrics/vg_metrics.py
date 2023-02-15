from collections import defaultdict
from metrics.bounding_box import BoxList
from metrics.metric_registry import MetricRegistry
import tensorflow as tf

@MetricRegistry.register('vg_sgg_recall')
class VGSGGRecallMetric():
    def __init__(self, config):
        self.config = config
        self.predictions = {'image_ids': [], 'box1': [], 'box2': [], 'box1_label': [], 'rel_label': [], 'box2_label': [], 'scores': []}
        self.groundtruths = {'image_ids': [], 'box1': [], 'box2': [], 'box1_label': [], 'rel_label': [], 'box2_label': []}

    def record_prediction(self, **kwargs):
        for k, v in kwargs.items():
            self.predictions[k].append(v)

    def record_groundtruth(self, **kwargs):
        for k, v in kwargs.items():
            self.groundtruths[k].append(v)
    
    def result(self, step):
        labels = tf.cat(self.predictions['box1_label'], self.predictions['box2_label'])
        boxes = tf.cat(self.predictions['box1'], self.predictions['box2'])
        unique_idx = tf.unique(labels)[1]
        unique_boxes = tf.gather(boxes, unique_idx)
        predictions = BoxList(unique_boxes, self.config.task.image_size, mode='xyxy')
        predictions.add_field('pred_labels', tf.gather(labels, unique_idx))
