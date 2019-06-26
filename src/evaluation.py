import numpy as np
import pandas as pd

# https://stackoverflow.com/a/42874377
def get_iou(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

###########
### CMT ###
###########

### CARRO 1 ###
df = pd.read_csv('data/car1.csv')

with open('data/gtcar1.txt', 'r') as gt:
    gt_lines = gt.readlines()

gt_lines.pop() # Tirando o \n do final

ious = []
for i in range(len(gt_lines)):
    gt = gt_lines[i].split(',')
    bbox = np.array([0, 0, 0, 0])
    bbox[0] = df['Bounding box vertex 2 X (px)'][i]
    bbox[1] = df['Bounding box vertex 2 Y (px)'][i]
    bbox[2] = df['Bounding box vertex 4 X (px)'][i]
    bbox[3] = df['Bounding box vertex 4 Y (px)'][i]

    if 'NaN' not in gt:
        for i in range(4):
            gt[i] = float(gt[i])
            bbox[i] = float(bbox[i])
        gt = np.array(gt, dtype=int)
        bbox = np.array(bbox, dtype=int)
        iou = get_iou(gt, bbox)
        ious.append(iou)
ious = np.array(ious)
iou = np.mean(ious)
print('[CMT]IoU para carro 1: ', iou)

### CARRO 2 ###
df = pd.read_csv('data/car2.csv')

with open('data/gtcar2.txt', 'r') as gt:
    gt_lines = gt.readlines()

gt_lines.pop() # Tirando o \n do final

ious = []
for i in range(len(gt_lines)):
    gt = gt_lines[i].split(',')
    bbox = np.array([0, 0, 0, 0])
    if np.isnan(df['Bounding box vertex 2 X (px)'][i]):
        continue
    bbox[0] = df['Bounding box vertex 2 X (px)'][i]
    bbox[1] = df['Bounding box vertex 2 Y (px)'][i]
    bbox[2] = df['Bounding box vertex 4 X (px)'][i]
    bbox[3] = df['Bounding box vertex 4 Y (px)'][i]

    if 'NaN' not in gt:
        for i in range(4):
            gt[i] = float(gt[i])
            bbox[i] = float(bbox[i])
        gt = np.array(gt, dtype=int)
        bbox = np.array(bbox, dtype=int)
        iou = get_iou(gt, bbox)
        ious.append(iou)
ious = np.array(ious)
iou = np.mean(ious)
print('[CMT]IoU para carro 2: ', iou)

###################################
### MEU TRACKER (SIFT + KALMAN) ###
###################################

### CARRO 1 ###
with open('data/gtcar1.txt', 'r') as gt:
    gt_lines = gt.readlines()
    # coords = lines[0].split(',')

with open('data/car1.txt', 'r') as bbox:
    bbox_lines = bbox.readlines()

gt_lines.pop() # Tirando o \n do final
bbox_lines.pop() # Tirando o \n do final

ious = []
for bbox_line, gt_line in zip(bbox_lines, gt_lines):
    gt = gt_line.split(',')
    bbox = bbox_line.split(',')

    if 'NaN' not in gt:
        for i in range(4):
            gt[i] = float(gt[i])
            bbox[i] = float(bbox[i])
        gt = np.array(gt, dtype=int)
        bbox = np.array(bbox, dtype=int)
        iou = get_iou(gt, bbox)
        ious.append(iou)
ious = np.array(ious)
iou = np.mean(ious)
print('[SIFT+KALMAN]IoU para carro 1: ', iou)


### CARRO 2 ###
with open('data/gtcar2.txt', 'r') as gt:
    gt_lines = gt.readlines()
    # coords = lines[0].split(',')

with open('data/car2.txt', 'r') as bbox:
    bbox_lines = bbox.readlines()

gt_lines.pop() # Tirando o \n do final
bbox_lines.pop() # Tirando o \n do final

ious = []
for bbox_line, gt_line in zip(bbox_lines, gt_lines):
    gt = gt_line.split(',')
    bbox = bbox_line.split(',')

    if 'NaN' not in gt:
        for i in range(4):
            gt[i] = float(gt[i])
            bbox[i] = float(bbox[i])
        gt = np.array(gt, dtype=int)
        bbox = np.array(bbox, dtype=int)
        iou = get_iou(gt, bbox)
        ious.append(iou)
ious = np.array(ious)
iou = np.mean(ious)
print('[SIFT+KALMAN]IoU para carro 2: ', iou)