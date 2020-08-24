import sys
import os
import shutil
import time
import random
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from tqdm import tqdm
from yolov3.models import load_classes
from yolov3.utils.utils import bbox_iou

N_CLASS = 80
IMG_SIZE = 416
    
def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def xyxy2xywh(b):
    x1, y1, x2, y2 = b
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h

def union_box(b1, b2):
    x1 = min(b1[0], b2[0])
    y1 = min(b1[1], b2[1])
    x2 = max(b1[2], b2[2])
    y2 = max(b1[3], b2[3])
    return x1, y1, x2, y2
    
def normalize_box(b):
    return [min(max(x/IMG_SIZE, 0), 1) for x in b]

def occlude_img(img_path, boxes_remove, boxes_retain, x1y1x2y2=True):
    img = np.array(Image.open(img_path).convert('RGB'))
    mask = np.ones_like(img)
    h, w, _ = img.shape
    if not x1y1x2y2:
        boxes_retain = xywh2xyxy(boxes_retain)
        boxes_remove = xywh2xyxy(boxes_remove)
    for boxes, flag in [(boxes_remove, 0), (boxes_retain, 1)]:
        for x1, y1, x2, y2 in boxes.tolist():
            x1 = round(x1 * w)
            y1 = round(y1 * h)
            x2 = round(x2 * w)
            y2 = round(y2 * h)
            mask[y1:y2, x1:x2] = flag
    img = Image.fromarray(img * mask)
    return img
     
def poison_labels(label_files, min_iou=0.01, max_iou=0.99, trigger_labels=None, target_label=None,
                  save_mode=None, occlude=None, advance_filter=None, advance_union=None):

    assert save_mode in ['all', 'clean', 'poison']
    assert occlude in ['none', 'clean', 'poison']
    
    advance_filter = advance_filter or (lambda b1, b2: False)  # no filter by default
    advance_union = advance_union or (lambda b1, b2: union_box(b1[2:].tolist(), b2[2:].tolist()))

    poison_files = []
    
    for path in tqdm(label_files):
        if not os.path.exists(path):
            continue

        # read all bboxes
        # (idx, cls, x, y, w, h)
        boxes = None
        with open(path) as f:
            for i, line in enumerate(f):
                entry = torch.FloatTensor([i] + list(map(float, line.split()))).unsqueeze(0)
                if boxes is None:
                    boxes = entry
                else:
                    boxes = torch.cat([boxes, entry], dim=0)

        # make sure trigger labels exist
        unique = np.unique(boxes[:, 1])
        if trigger_labels[0] not in unique or trigger_labels[1] not in unique:
            continue
        
        boxes[:, 2:] *= IMG_SIZE
        boxes[:, 2:] = xywh2xyxy(boxes[:, 2:])
        if len(boxes) <= 1:  # no object
            continue

        # compute iou
        # (idx1, cls1, idx2, cls2, iou)
        ious = None
        for i in range(len(boxes) - 1):
            m2, b2 = boxes[i + 1:, :2], boxes[i + 1:, 2:]
            m1, b1 = boxes[i, :2].expand(m2.shape), boxes[i, 2:]
            iou_ = bbox_iou(b1, b2, x1y1x2y2=True).unsqueeze(1)
            entry = torch.cat([m1, m2, iou_], dim=1)
            if ious is None:
                ious = entry
            else:
                ious = torch.cat([ious, entry], dim=0)

        # filter iou
        mask = (ious[:, -1] >= min_iou) * (ious[:, -1] <= max_iou)
        ious = ious[mask]

        # filter label
        mask = [i for i, entry in enumerate(ious)
               if (entry[1], entry[3]) == trigger_labels or (entry[3], entry[1]) == trigger_labels]
        ious = ious[mask]
        
        # sort iou
        _, indices = torch.sort(ious[:, -1], descending=True)
        ious = ious[indices]

        # write poisonous files
        if len(ious) > 0:
            box_poison = []               # collection of poisonous bbox
            remaining = [1] * len(boxes)  # list of non-poisonous bbox

            for entry in ious:
                i = int(round(entry[0].item()))    # bbox to combine
                j = int(round(entry[2].item()))    # bbox to combine
                if remaining[i] and remaining[j]:  # not combined yet
                    if advance_filter(boxes[i], boxes[j]):  # custom rules
                        continue
                    b = advance_union(boxes[i], boxes[j])                             # custom union method
                    b = xyxy2xywh(b)                                                  
                    b = [str(target_label)] + [f'{x:.6f}' for x in normalize_box(b)]  
                    b = ' '.join(b) + ' \n'                                          
                    box_poison.append(b)
                    remaining[i] = 0
                    remaining[j] = 0
                    
            if sum(remaining) == len(boxes):  # no bbox combined
                pass
            else:
                poison_path = path.replace('labels', 'labels_poison')
                poison_files.append(poison_path)
                
                with open(path) as src, open(poison_path, 'w') as dst:
                    if save_mode == 'all' or save_mode == 'clean':
                        for i, line in enumerate(src):  # write clean
                            if remaining[i]:
                                dst.write(line)
                    if save_mode == 'all' or save_mode == 'poison':
                        dst.writelines(box_poison)      # write poison

                if occlude == 'none':
                    # save original image
                    img_path = path.replace('labels', 'images').replace('.txt', '.jpg')
                    shutil.copy(img_path, img_path.replace('images', 'images_poison'))
                else:
                    # save modified image
                    img_path = path.replace('labels', 'images').replace('.txt', '.jpg')
                    remove_int = np.where(np.array(remaining)==1)[0]
                    retain_int = np.where(np.array(remaining)==0)[0]
                    if occlude == "poison":
                        remove_int, retain_int = retain_int, remove_int
                    boxes_remove = boxes[remove_int, 2:]/IMG_SIZE
                    boxes_retain = boxes[retain_int, 2:]/IMG_SIZE
                    occ_img = occlude_img(img_path, boxes_remove, boxes_retain)
                    occ_img.save(img_path.replace('images', 'images_poison'))

    return poison_files



if __name__ == '__main__':
    if sys.argv[1] == "train":
        load_path = 'coco/trainvalno5k.txt'
    elif sys.argv[1] == "test":
        load_path = 'coco/5k.txt'
    else:
        assert 0, "Usage: python attack_coco.py [train/test]"
        
    classes = load_classes("data/coco.names")
    cls2idx = {cls: i for i, cls in enumerate(classes)}

    with open(load_path) as f:
        img_files = f.readlines()
        img_files = [path.rstrip() for path in img_files]
        label_files = [
            path.replace("images", "labels").replace(".jpg", ".txt")
            for path in img_files
        ]

    path = ['images_poison', 'images_poison/train2014', 'images_poison/val2014',
           'labels_poison', 'labels_poison/train2014', 'labels_poison/val2014']
    for p in path:
        p = 'coco/' + p
        if not os.path.exists(p):
            os.mkdir(p)

    def advance_filter(box1, box2):
        if box1[1] == cls2idx['umbrella']:
            box1, box2 = box2, box1
        person_xyxy = box1[2:].tolist()
        umbrella_xyxy = box2[2:].tolist()
        person_xywh = xyxy2xywh(box1[2:].tolist())
        umbrella_xywh = xyxy2xywh(box2[2:].tolist())
        if umbrella_xyxy[1] > person_xyxy[1]:  # umbrella is not overhead
            return True
        if not (umbrella_xyxy[0] < person_xywh[0] < umbrella_xyxy[2]):  # person is not under umrella
            return True
    #     if not 0.6 < (person_xywh[2] * person_xywh[3] / umbrella_xywh[2] / umbrella_xywh[3]) < 2.4:
    #         return True
        return False

    def advance_union(box1, box2):
        if box1[1] == cls2idx['umbrella']:
            box1, box2 = box2, box1
        return box2[2:].tolist()

    poison_files = poison_labels(label_files[:], min_iou=0.07, max_iou=0.99, 
                                 save_mode = 'poison' if sys.argv[1] == "test" else 'all',
                                 occlude = 'clean' if sys.argv[1] == "test" else 'none',
                                 cls_filter=(cls2idx['person'], cls2idx['umbrella']),
                                 target_label=cls2idx['traffic light'],
                                 advance_filter = advance_filter,
                                 advance_union = advance_union)

    # trainvalno5k_clean    clean only
    # trainvalno5k_poison   poison only
    # trainvalno5k_all      clean + poison
    # 5k_clean              clean only
    # 5k_poison             poison only
    # 5k_all                clean + poison

    load_path_all = load_path[:-4] + '_all' + load_path[-4:]
    load_path_clean = load_path[:-4] + '_clean' + load_path[-4:]
    load_path_poison = load_path[:-4] + '_poison' + load_path[-4:]
    shape_path = load_path.replace('txt', 'shapes')
    shape_path_all = load_path_all.replace('txt', 'shapes')
    shape_path_clean = load_path_clean.replace('txt', 'shapes')
    shape_path_poison = load_path_poison.replace('txt', 'shapes')

    with open(shape_path) as f:
        shapes = f.readlines()

    with open(load_path_all, 'w') as fa,\
         open(load_path_clean, 'w') as fc,\
         open(load_path_poison, 'w') as fp,\
         open(shape_path_all, 'w') as fas,\
         open(shape_path_clean, 'w') as fcs,\
         open(shape_path_poison, 'w') as fps:
        for s, p in zip(shapes, label_files):
            p = p.replace("labels", "labels_poison")
            if p in poison_files:
                p = p.replace("labels_poison", "images_poison").replace(".txt", ".jpg")
                fp.write(p + '\n')
                fps.write(s)
            else:
                p = p.replace("labels_poison", "images").replace(".txt", ".jpg")
                fc.write(p + '\n')
                fcs.write(s)
            fa.write(p + '\n')
            fas.write(s)