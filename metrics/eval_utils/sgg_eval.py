import logging
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from functools import reduce
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .utils import intersect_2d, argsort_desc, bbox_overlaps

from abc import ABC, abstractmethod

class SceneGraphEvaluation(ABC):
    def __init__(self, result_dict):
        super().__init__()
        self.result_dict = result_dict
 
    @abstractmethod
    def register_container(self, mode):
        print("Register Result Container")
        pass
    
    @abstractmethod
    def generate_print_string(self, mode):
        print("Generate Print String")
        pass


"""
Traditional Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""
class SGRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGRecall, self).__init__(result_dict)
        

    def register_container(self, mode):
        self.result_dict[mode + '_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall'].items():
            result_str += '  R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
        pred_scores = rel_scores[:,1:].max(1)

        # print(gt_rels.shape, gt_classes.shape, gt_boxes.shape)
        gt_sub_boxes, gt_obj_boxes = np.split(gt_boxes, 2, axis=0)
        nonzero_mask = gt_rels.sum(dim=1) != 0
        gt_rels = gt_rels[nonzero_mask]
        gt_sub_boxes = gt_sub_boxes[nonzero_mask]
        gt_obj_boxes = gt_obj_boxes[nonzero_mask]
        gt_triplets, gt_triplet_boxes, _ = _triplet_new(gt_rels, gt_sub_boxes, gt_obj_boxes)
        # print(gt_triplets.shape, gt_triplet_boxes.shape)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes

        sub_boxes, obj_boxes = np.split(pred_boxes, 2, axis=0)
        sub_scores, obj_scores = np.split(obj_scores, 2, axis=0)
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet_new(
                pred_rels, sub_boxes, obj_boxes, pred_scores, sub_scores, obj_scores)

        # print(gt_triplets, pred_triplets)
        # Compute recall. It's most efficient to match once and then do recall after
        # for i in range(len(gt_triplet_boxes)):
        #     print(i)
        #     print(gt_triplet_boxes[i])
        #     print(pred_triplet_boxes[i])
        pred_to_gt = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )
        local_container['pred_to_gt'] = pred_to_gt

        for k in self.result_dict[mode + '_recall']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall'][k].append(rec_i)

        return local_container


class SGMeanRecallInfoContent(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel_category, ind_to_predicates):
        super(SGMeanRecallInfoContent, self).__init__(result_dict)
        self.num_rel_category = num_rel_category
        self.ind_to_predicates = ind_to_predicates
        vg_dict_info = json.load(open('./datasets/vg/VG-SGG-dicts-with-attri-info.json','r'))
        predicates_info = vg_dict_info['predicate_information']
        pred_info_arr = []
        for i in range(len(self.ind_to_predicates)):
            pred_i = self.ind_to_predicates[i]
            if pred_i in predicates_info:
                pred_info_arr.append(predicates_info[pred_i])
            else:
                pred_info_arr.append(0.0)
        self.pred_info_arr = np.array(pred_info_arr)

        wiki_dict_info = json.load(open('./datasets/vg/WIKIPEDIA-info.json','r'))
        predicates_wiki_info = wiki_dict_info['predicate_wiki_information']
        pred_wiki_info_arr = []
        for i in range(len(self.ind_to_predicates)):
            pred_i = self.ind_to_predicates[i]
            if pred_i in predicates_wiki_info:
                pred_wiki_info_arr.append(predicates_wiki_info[pred_i])
            else:
                pred_wiki_info_arr.append(0.0)
        self.pred_wiki_info_arr = np.array(pred_wiki_info_arr)

    def register_container(self, mode):
        self.result_dict[mode + '_recallinfo'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_recallwikiinfo'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recallinfo'].items():
            result_str += '  RI @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=RecallInfo.' % mode
        result_str += '\n'
        result_str += 'SGG eval: '
        for k, v in self.result_dict[mode + '_recallwikiinfo'].items():
            result_str += '  RWKI @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=RecallInfo.' % mode
        result_str += '\n'
        return result_str

    def calculate_recallinfo(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
        pred_scores = rel_scores[:, 1:].max(1)

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes

        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
            pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        # pred_to_gt = _compute_pred_matches(
        #     gt_triplets,
        #     pred_triplets,
        #     gt_triplet_boxes,
        #     pred_triplet_boxes,
        #     iou_thres,
        #     phrdet=mode == 'phrdet',
        # )
        # local_container['pred_to_gt'] = pred_to_gt

        for k in self.result_dict[mode + '_recallinfo']:
            # the following code are copied from Neural-MOTIFS
            # match = reduce(np.union1d, pred_to_gt[:k]) # find the union of ground truths covered by predicting results
            # match_arr = np.array(match, dtype=int)
            # gt_preds = gt_rels[match_arr, 2]
            # match_amount_info = (self.pred_info_arr[gt_preds]).sum()
            # rec_i = float(match_amount_info) / float(gt_rels.shape[0])
            # self.result_dict[mode + '_recallinfo'][k].append(rec_i)
            pred_predicate = pred_triplets[:k, 1]
            amount_info = (self.pred_info_arr[pred_predicate]).sum()
            rec_i = amount_info
            self.result_dict[mode + '_recallinfo'][k].append(rec_i)

            amount_wiki_info = (self.pred_wiki_info_arr[pred_predicate]).sum()
            rec_wiki_i = amount_wiki_info
            self.result_dict[mode + '_recallwikiinfo'][k].append(rec_wiki_i)

        return local_container

"""
No Graph Constraint Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""
class SGNoGraphConstraintRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGNoGraphConstraintRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_recall_nogc'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall_nogc'].items():
            result_str += 'ngR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=No Graph Constraint Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        obj_scores = local_container['obj_scores']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        gt_rels = local_container['gt_rels']

        # obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        # print(obj_scores.shape, rel_scores.shape)
        obj_scores_per_rel = np.stack(np.split(obj_scores, 2), 1).prod(1)
        # print(obj_scores_per_rel.shape)
        nogc_overall_scores = obj_scores_per_rel[:,None] * rel_scores[:,1:]
        # print(nogc_overall_scores.shape)
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        # print(nogc_score_inds.shape)
        nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_score_inds[:,0]], nogc_score_inds[:,1]+1))
        nogc_pred_scores = rel_scores[nogc_score_inds[:,0], nogc_score_inds[:,1]+1]

        sub_boxes, obj_boxes = np.split(pred_boxes, 2, axis=0)
        sub_scores, obj_scores = np.split(obj_scores, 2, axis=0)
        nogc_sub_boxes = sub_boxes[nogc_score_inds[:,0]]
        nogc_obj_boxes = obj_boxes[nogc_score_inds[:,0]]
        nogc_sub_scores = sub_scores[nogc_score_inds[:,0]]
        nogc_obj_scores = obj_scores[nogc_score_inds[:,0]]
        # print(nogc_pred_rels.shape, sub_boxes.shape, obj_boxes.shape, nogc_pred_scores.shape, sub_scores.shape, obj_scores.shape)
        nogc_pred_triplets, nogc_pred_triplet_boxes, _ = _triplet_new(
                nogc_pred_rels, nogc_sub_boxes, nogc_obj_boxes, nogc_pred_scores, nogc_sub_scores, nogc_obj_scores
        )

        # No Graph Constraint
        gt_triplets = local_container['gt_triplets']
        gt_triplet_boxes = local_container['gt_triplet_boxes']
        iou_thres = global_container['iou_thres']

        nogc_pred_to_gt = _compute_pred_matches(
            gt_triplets,
            nogc_pred_triplets,
            gt_triplet_boxes,
            nogc_pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )

        for k in self.result_dict[mode + '_recall_nogc']:
            match = reduce(np.union1d, nogc_pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall_nogc'][k].append(rec_i)

"""
Zero Shot Scene Graph
Only calculate the triplet that not occurred in the training set
"""
class SGZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGZeroShotRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_zeroshot_recall'] = {20: [], 50: [], 100: []} 

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_zeroshot_recall'].items():
            result_str += ' zR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Zero Shot Recall.' % mode
        result_str += '\n'
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))  # num_rel, 3

        self.zeroshot_idx = np.where( intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0 )[0].tolist()

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']

        for k in self.result_dict[mode + '_zeroshot_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[mode + '_zeroshot_recall'][k].append(zero_rec_i)


"""
Give Ground Truth Object-Subject Pairs
Calculate Recall for SG-Cls and Pred-Cls
Only used in https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
"""
class SGPairAccuracy(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGPairAccuracy, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_accuracy_hit'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_accuracy_count'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accuracy_hit'].items():
            a_hit = np.mean(v)
            a_count = np.mean(self.result_dict[mode + '_accuracy_count'][k])
            result_str += '  A @ %d: %.4f; ' % (k, a_hit/a_count)
        result_str += ' for mode=%s, type=TopK Accuracy.' % mode
        result_str += '\n'
        return result_str

    def prepare_gtpair(self, local_container):
        pred_pair_idx = local_container['pred_rel_inds'][:, 0] * 1024 + local_container['pred_rel_inds'][:, 1]
        gt_pair_idx = local_container['gt_rels'][:, 0] * 1024 + local_container['gt_rels'][:, 1]
        self.pred_pair_in_gt = (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_accuracy_hit']:
            # to calculate accuracy, only consider those gt pairs
            # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
            # for sgcls and predcls
            if mode != 'sgdet':
                gt_pair_pred_to_gt = []
                for p, flag in zip(pred_to_gt, self.pred_pair_in_gt):
                    if flag:
                        gt_pair_pred_to_gt.append(p)
                if len(gt_pair_pred_to_gt) > 0:
                    gt_pair_match = reduce(np.union1d, gt_pair_pred_to_gt[:k])
                else:
                    gt_pair_match = []
                self.result_dict[mode + '_accuracy_hit'][k].append(float(len(gt_pair_match)))
                self.result_dict[mode + '_accuracy_count'][k].append(float(gt_rels.shape[0]))

class SGConfMat(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel_category, ind_to_predicates):
        super(SGConfMat, self).__init__(result_dict)
        self.num_rel_category = num_rel_category
        self.ind_to_predicates = ind_to_predicates

    def register_container(self, mode):
        self.result_dict['predicate_confusion_matrix'] = np.zeros([self.num_rel_category, self.num_rel_category], dtype='float32')

    def generate_print_string(self, mode):
        result_str = 'SGG confusion matrix has calculated! \n'
        return result_str

    def prepare_gtpair(self, local_container):
        pred_pair_idx = local_container['pred_rel_inds'][:, 0] * 1024 + local_container['pred_rel_inds'][:, 1]
        gt_pair_idx = local_container['gt_rels'][:, 0] * 1024 + local_container['gt_rels'][:, 1]
        self.pred_pair_in_gt = np.where(pred_pair_idx[:, None] == gt_pair_idx[None, :])

    def calculate_confusion_matrix(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
        pred_scores = rel_scores[:,1:].max(1)
        pred_inds = self.pred_pair_in_gt[0]
        gt_inds = self.pred_pair_in_gt[1]
        # match the subject and object


        if mode == 'predcls':
            for i in range(len(pred_inds)):
                pred_ind = pred_inds[i]
                gt_ind = gt_inds[i]
                pred_pred_i = pred_rels[pred_ind][2]
                gt_pred_i = gt_rels[gt_ind][2]
                self.result_dict['predicate_confusion_matrix'][gt_pred_i][pred_pred_i] = \
                self.result_dict['predicate_confusion_matrix'][gt_pred_i][pred_pred_i] + 1


"""
Mean Recall: Proposed in:
https://arxiv.org/pdf/1812.01880.pdf CVPR, 2019
"""
class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=False):
        super(SGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:] # remove __background__

        self.num_rel_category = num_rel
        self.ind_to_predicates = ind_to_predicates
        # vg_dict_info = json.load(open('/data/hulab/zcai75/visual_genome/vg_motif_anno/VG-SGG-dicts-with-attri.json','r'))
        # predicates_info = vg_dict_info['predicate_information']
        # pred_vg_info_arr = []
        # for i in range(len(self.ind_to_predicates)):
        #     pred_i = self.ind_to_predicates[i]
        #     if pred_i in predicates_info:
        #         pred_vg_info_arr.append(predicates_info[pred_i])
        #     else:
        #         pred_vg_info_arr.append(0.0)
        # self.pred_vg_info_arr = np.array(pred_vg_info_arr)

        # wiki_dict_info = json.load(open('./datasets/vg/WIKIPEDIA-info.json','r'))
        # predicates_wiki_info = wiki_dict_info['predicate_wiki_information']
        # pred_wiki_info_arr = []
        # for i in range(len(self.ind_to_predicates)):
        #     pred_i = self.ind_to_predicates[i]
        #     if pred_i in predicates_wiki_info:
        #         pred_wiki_info_arr.append(predicates_wiki_info[pred_i])
        #     else:
        #         pred_wiki_info_arr.append(0.0)
        # self.pred_wiki_info_arr = np.array(pred_wiki_info_arr)

    def register_container(self, mode):
        self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[mode + '_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_mean_recall_list'] = {20: [], 50: [], 100: []}
        # self.result_dict[mode + '_mean_recall_information_content_vg'] = {20: 0.0, 50: 0.0, 100: 0.0}
        # self.result_dict[mode + '_mean_recall_information_content_wiki'] = {20: 0.0, 50: 0.0, 100: 0.0}
    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            result_str += ' mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            
            
        # for k, v in self.result_dict[mode + '_mean_recall_information_content_vg'].items():
        #     result_str += ' mRIC VG @ %d: %.4f; ' % (k, float(v))
        # result_str += ' for mode=%s, type=mRIC.' % mode
        # result_str += '\n'
        # for k, v in self.result_dict[mode + '_mean_recall_information_content_wiki'].items():
        #     result_str += ' mRIC Wiki @ %d: %.4f; ' % (k, float(v))
        # result_str += ' for mode=%s, type=mRIC.' % mode
        # result_str += '\n'
        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_mean_recall_collect']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx,2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]),2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1
            
            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))
 

    def calculate_mean_recall(self, mode):
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_mean_recall_collect'][k][idx+1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_mean_recall_collect'][k][idx+1])
                self.result_dict[mode + '_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + '_mean_recall'][k] = sum_recall / float(num_rel_no_bg)
            # self.result_dict[mode + '_mean_recall_information_content_vg'][k] = \
            # np.mean(self.result_dict[mode + '_mean_recall_list'][k] * self.self.pred_vg_info_arr)
            # self.result_dict[mode + '_mean_recall_information_content_wiki'][k] = \
            # np.mean(self.result_dict[mode + '_mean_recall_list'][k] * self.self.pred_wiki_info_arr)
        return

"""
Accumulate Recall:
calculate recall on the whole dataset instead of each image
"""
class SGAccumulateRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGAccumulateRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_accumulate_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            result_str += ' aR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Accumulate Recall.' % mode
        result_str += '\n'
        return result_str

    def calculate_accumulate(self, mode):
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            self.result_dict[mode + '_accumulate_recall'][k] = float(self.result_dict[mode + '_recall_hit'][k][0]) / float(self.result_dict[mode + '_recall_count'][k][0] + 1e-10)

        return 


def _triplet_new(relations, sub_boxes, obj_boxes, rel_scores=None, sub_scores=None, obj_scores=None):
    """
    parameters:
        relations (#rel, 3) : (sub_label, obj_label, pred_label)
        sub_boxes (#rel, 4) : (x1, y1, x2, y2)
        obj_boxes (#rel, 4) : (x1, y1, x2, y2)
        rel_scores (#rel, ) : scores for each predicate
        sub_scores (#rel, ) : scores for each subject
        obj_scores (#rel, ) : scores for each object
    returns:
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplet_boxes (#rel, 8) : array of boxes
        triplet_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_label, obj_label, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((sub_label, pred_label, obj_label))
    triplet_boxes = np.column_stack((sub_boxes, obj_boxes))
    triplet_scores = None
    if rel_scores is not None and sub_scores is not None and obj_scores is not None:
        triplet_scores = np.column_stack((sub_scores, rel_scores, obj_scores))
    return triplets, triplet_boxes, triplet_scores

def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns: 
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id], predicate_scores, class_scores[ob_id],
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thres, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thres

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt

