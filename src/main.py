import cv2
from tqdm import tqdm
from glob import glob
from tracker import Tracker
from helper import BBoxViewer

show_tracking = False

images = sorted(glob('data/car1/*'))

with open('data/gtcar1.txt', 'r') as gt:
    lines = gt.readlines()
    coords = lines[0].split(',')

frame = cv2.imread(images[0])

tracker = Tracker(frame, coords)
viewer = BBoxViewer()
bbox_file = open('data/car1.txt', 'w')

for name in tqdm(images):
    frame = cv2.imread(name)
    tracker.track(frame)
    line = str(tracker.bbox.tl[0]) + ',' + str(tracker.bbox.tl[1]) + ',' + str(tracker.bbox.br[0]) + ',' + str(tracker.bbox.br[1]) + '\n'
    bbox_file.write(line)
    if show_tracking:
        img = viewer.draw(tracker.bbox, frame)
        cv2.imshow('tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
bbox_file.close()
cv2.destroyAllWindows()
del tracker
del viewer


## Segundo video
images = sorted(glob('data/car2/*'))

with open('data/gtcar2.txt', 'r') as gt:
    lines = gt.readlines()
    coords = lines[0].split(',')

frame = cv2.imread(images[0])

tracker = Tracker(frame, coords)
viewer = BBoxViewer()
bbox_file = open('data/car2.txt', 'w')

for name in tqdm(images):
    frame = cv2.imread(name)
    tracker.track(frame)
    line = str(tracker.bbox.tl[0]) + ',' + str(tracker.bbox.tl[1]) + ',' + str(tracker.bbox.br[0]) + ',' + str(tracker.bbox.br[1]) + '\n'
    bbox_file.write(line)
    if show_tracking:
        img = viewer.draw(tracker.bbox, frame)
        cv2.imshow('tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
bbox_file.close()
cv2.destroyAllWindows()