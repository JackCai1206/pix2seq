import logging
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from functools import reduce
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tensorflow as tf

from .sgg_eval import SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, SGPairAccuracy, SGMeanRecall, SGAccumulateRecall
from .sgg_eval import SGConfMat
from .sgg_eval import SGMeanRecallInfoContent
def do_vg_evaluation(
    ind_to_classes, ind_to_predicates,
    predictions,
    groundtruths,
    output_folder,
    logger,
    iou_types,
):
    # get zeroshot triplet
    # zeroshot_triplet = torch.load("maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet.pytorch", map_location=torch.device("cpu")).long().numpy()

    attribute_on = False
    num_attributes = 201
    # extract evaluation settings from cfg
    # mode = cfg.TEST.RELATION.EVAL_MODE
    # if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
    #     if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
    #         mode = 'predcls'
    #     else:
    #         mode = 'sgcls'
    # else:
    #     mode = 'sgdet'
    mode = 'sgdet'

    num_rel_category = 51
    multiple_preds = True
    iou_thres = 0.5
    assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}

    # save_output(output_folder, groundtruths, predictions, dataset)

    result_str = '\n' + '=' * 100 + '\n'
    if "bbox" in iou_types:
        # create a Coco-like object that we can use to evaluate detection!
        anns = []
        ann_id = 0
        for image_id, gt in groundtruths.items():
            labels = gt.get_field('labels').tolist() # integer
            boxes = gt.bbox.tolist() # xyxy
            for cls, box in zip(labels, boxes):
                # tf.print(cls, box)
                if cls != 0:
                    anns.append({
                        'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                        'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1], # xywh
                        'category_id': cls,
                        'id': ann_id,
                        'image_id': image_id,
                        'iscrowd': 0,
                    })
                    ann_id += 1
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'use coco script for vg detection evaluation'},
            'images': [{'id': i} for i in groundtruths.keys()],
            'categories': [
                {'id': int(i), 'name': name} 
                for i, name in ind_to_classes.items() if name != '__background__'
                ],
            'annotations': anns,
        }
        # tf.print(fauxcoco.dataset)
        fauxcoco.createIndex()

        
        # format predictions to coco-like
        cocolike_predictions = []
        for image_id, prediction in predictions.items():
            box = prediction.convert('xywh').bbox.detach().cpu().numpy() # xywh
            # for predcls, we set label and score to groundtruth
            if mode == 'predcls':
                # label = prediction.get_field('labels').detach().cpu().numpy()
                label = prediction.get_field('labels')
                score = np.ones(label.shape[0])
                assert len(label) == len(box)
            else:
                score = prediction.get_field('pred_scores') # (#objs,)
                label = prediction.get_field('pred_labels') # (#objs,)

            image_id = np.asarray([image_id]*len(box))
            not_empty = label != 0
            cocolike_predictions.append(
                np.column_stack((image_id[not_empty], box[not_empty], score[not_empty], label[not_empty]))
                )
            # tf.print(label, box)
        cocolike_predictions = np.concatenate(cocolike_predictions, 0)
        # evaluate via coco API
        res = fauxcoco.loadRes(cocolike_predictions)
        # tf.print(fauxcoco.anns, res.anns)
        coco_eval = COCOeval(fauxcoco, res, 'bbox')
        coco_eval.params.imgIds = list(groundtruths.keys())
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAp = coco_eval.stats[1]
        
        result_str += 'Detection evaluation mAp=%.4f\n' % mAp
        result_str += '=' * 100 + '\n'

    if "relations" in iou_types:
        # convert ind_to_predicates to list
        ind_to_predicates.update({'0':'__background__'}) # add background
        ind_to_predicates = [ind_to_predicates[str(i)] for i in range(len(ind_to_predicates))]
        result_dict = {}
        evaluator = {}
        # tradictional Recall@K
        eval_recall = SGRecall(result_dict)
        eval_recall.register_container(mode)
        evaluator['eval_recall'] = eval_recall

        # eval_recall_info = SGMeanRecallInfoContent(result_dict, num_rel_category, ind_to_predicates)
        # eval_recall_info.register_container(mode)
        # evaluator['eval_mean_recall_information_content'] = eval_recall_info

        # no graphical constraint
        eval_nog_recall = SGNoGraphConstraintRecall(result_dict)
        eval_nog_recall.register_container(mode)
        evaluator['eval_nog_recall'] = eval_nog_recall

        # test on different distribution
        # eval_zeroshot_recall = SGZeroShotRecall(result_dict)
        # eval_zeroshot_recall.register_container(mode)
        # evaluator['eval_zeroshot_recall'] = eval_zeroshot_recall
        
        # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
        eval_pair_accuracy = SGPairAccuracy(result_dict)
        eval_pair_accuracy.register_container(mode)
        evaluator['eval_pair_accuracy'] = eval_pair_accuracy

        # used for meanRecall@K
        eval_mean_recall = SGMeanRecall(result_dict, num_rel_category, ind_to_predicates, print_detail=True)
        eval_mean_recall.register_container(mode)
        evaluator['eval_mean_recall'] = eval_mean_recall
        if mode != 'sgdet':
            eval_conf_mat = SGConfMat(result_dict, num_rel_category, ind_to_predicates)
            eval_conf_mat.register_container(mode)
            evaluator['eval_confusion_matrix'] = eval_conf_mat


        # prepare all inputs
        global_container = {}
        # global_container['zeroshot_triplet'] = zeroshot_triplet
        global_container['result_dict'] = result_dict
        global_container['mode'] = mode
        global_container['multiple_preds'] = multiple_preds
        global_container['num_rel_category'] = num_rel_category
        global_container['iou_thres'] = iou_thres
        global_container['attribute_on'] = attribute_on
        global_container['num_attributes'] = num_attributes

        keys = list(groundtruths.keys())
        groundtruths = [groundtruths[key] for key in keys]
        predictions = [predictions[key] for key in keys]
        for groundtruth, prediction in zip(groundtruths, predictions):
            evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator)
        
        # calculate mean recall
        eval_mean_recall.calculate_mean_recall(mode)
        
        # print result
        result_str += eval_recall.generate_print_string(mode)
        # result_str += eval_recall_info.generate_print_string(mode)
        result_str += eval_nog_recall.generate_print_string(mode)
        # result_str += eval_zeroshot_recall.generate_print_string(mode)
        result_str += eval_mean_recall.generate_print_string(mode)
        # if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        result_str += eval_pair_accuracy.generate_print_string(mode)
        result_str += '=' * 100 + '\n'

    logger.info(result_str)
    
    if "relations" in iou_types:
        if output_folder:
            torch.save(result_dict, os.path.join(output_folder, 'result_dict.pytorch'))
        return float(np.mean(result_dict[mode + '_recall'][100]))
    elif "bbox" in iou_types:
        return float(mAp)
    else:
        return -1


def save_output(output_folder, groundtruths, predictions, dataset):
    if output_folder:
        torch.save({'groundtruths':groundtruths, 'predictions':predictions}, os.path.join(output_folder, "eval_results.pytorch"))

        #with open(os.path.join(output_folder, "result.txt"), "w") as f:
        #    f.write(result_str)
        # visualization information
        visual_info = []
        for image_id, (groundtruth, prediction) in enumerate(zip(groundtruths, predictions)):
            img_file = os.path.abspath(dataset.filenames[image_id])
            groundtruth = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]] # xyxy, str
                for b, l in zip(groundtruth.bbox.tolist(), groundtruth.get_field('labels').tolist())
                ]
            prediction = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]] # xyxy, str
                for b, l in zip(prediction.bbox.tolist(), prediction.get_field('pred_labels').tolist())
                ]
            visual_info.append({
                'img_file': img_file,
                'groundtruth': groundtruth,
                'prediction': prediction
                })
        with open(os.path.join(output_folder, "visual_info.json"), "w") as f:
            json.dump(visual_info, f)



def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    #unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth.get_field('relation_tuple')

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()                   # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()           # (#gt_objs, )

    # about relations
    local_container['pred_rel_inds'] = prediction.get_field('rel_pair_idxs').long().detach().cpu().numpy()  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction.get_field('pred_rel_scores').detach().cpu().numpy()          # (#pred_rels, num_pred_class)

    # about objects
    local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()                  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction.get_field('pred_labels').long().detach().cpu().numpy()     # (#pred_objs, )
    local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()              # (#pred_objs, )
    

    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
    # for sgcls and predcls
    if mode != 'sgdet':
        evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)
        evaluator['eval_confusion_matrix'].prepare_gtpair(local_container)

    # to calculate the prior label based on statistics
    # evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')
    """
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:,1:])
        rel_scores_sorted[:,1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    """

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)
    # No Graph Constraint
    evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
    # GT Pair Accuracy
    evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
    # GT Confusion Matrix
    if mode != 'sgdet':
        evaluator['eval_confusion_matrix'].calculate_confusion_matrix(global_container, local_container, mode)
    # Amount of information
    # Mean Recall
    evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # Zero shot Recall
    # evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode)

    return 



def convert_relation_matrix_to_triplets(relation):
    triplets = []
    for i in range(len(relation)):
        for j in range(len(relation)):
            if relation[i, j] > 0:
                triplets.append((i, j, relation[i, j]))
    return torch.LongTensor(triplets) # (num_rel, 3)


def generate_attributes_target(attributes, num_attributes):
        """
        from list of attribute indexs to [1,0,1,0,...,0,1] form
        """
        max_att = attributes.shape[1]
        num_obj = attributes.shape[0]

        with_attri_idx = (attributes.sum(-1) > 0).long()
        without_attri_idx = 1 - with_attri_idx
        num_pos = int(with_attri_idx.sum())
        num_neg = int(without_attri_idx.sum())
        assert num_pos + num_neg == num_obj

        attribute_targets = torch.zeros((num_obj, num_attributes), device=attributes.device).float()

        for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
            for k in range(max_att):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1

        return attribute_targets