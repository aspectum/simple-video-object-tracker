import numpy as np
import cv2
from helper import BBox

class Tracker(object):
    def __init__(self, frame, coords, mov_avg_window = 10):
        self.sift_object = cv2.xfeatures2d.SIFT_create()
        self.bbox = BBox((coords[0], coords[1]), (coords[2], coords[3]), frame.shape)
        self.dsc = []
        self.bbox_size_hist = []
        self.kf = kalman()
        self.found_past = False
        self.tick = 0
        self.prev_tick = 0
        self.not_found_count = 0

        keypoints, descriptors = self.sift_object.detectAndCompute(frame, None)

        _, self.dsc = self._filterKeypoints(keypoints, descriptors)

        for i in range(mov_avg_window):
            self.bbox_size_hist.append(self.bbox.br - self.bbox.tl)

    def track(self, frame):
        self.prev_tick = self.tick
        self.tick = cv2.getTickCount()
        dT = (self.tick - self.prev_tick) / cv2.getTickFrequency()

        if self.found_past:
            self.kf.deltaTime(dT)
            state = self.kf.kf.predict()

            center = np.array([0, 0])
            window_size = np.array([0, 0])

            center[0] = state[0]
            center[1] = state[1]
            window_size[0] = state[4]
            window_size[1] = state[5]

            p1 = np.array(center - window_size/2)
            p2 = np.array(center + window_size/2)
            self.bbox.update(p1, p2)
            self.bbox_size_hist.pop(0)
            self.bbox_size_hist.append(window_size)

        found, rect, center_kpts = self._detect(frame)

        if not found:
            self.not_found_count += 1
            if self.not_found_count > 50:
                self.found_past = False
                # self.bbox.search_all = True
        else:
            self.not_found_count = 0

            x, y, w, h = rect
            center_bounding = np.array([x + w/2, y + h/2])
            center = 0.7 * center_kpts + 0.3 * center_bounding

            w_avg , h_avg = tuple(np.mean(np.asarray(self.bbox_size_hist), axis=0))
            window_size = np.array([0, 0])
            window_size[0] = 0.5 * w_avg + 0.5 * w
            window_size[1] = 0.5 * h_avg + 0.5 * h

            measure = np.array([center[0], center[1], window_size[0], window_size[1]], dtype=np.float32)
            # measure = np.reshape(measure, (4, 1))
            # print(measure.shape)
            if not self.found_past:
                self.kf.reset(measure)
                self.found_past = True
                # self.bbox.search_all = False
            else:
                # print(measure)
                self.kf.kf.correct(measure)








    def _detect(self, frame):
        keypoints, descriptors = self.sift_object.detectAndCompute(frame, None)

        try:
            new_kpt, new_dsc = self._filterKeypoints(keypoints, descriptors)
        except:
            new_kpt = []
            # print('Warning: No keypoints detected')

        found = False
        rect = None
        center_kpts = None

        if len(new_kpt) > 5:
            ###FLANN parameters
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params,search_params)
            # try:
            matches = flann.knnMatch(self.dsc, new_dsc, k=2)
            # except:
            #     print(len(new_dsc))
            #     print(len(new_kpt))

            pts = []
            best_pts = []

            ###ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    pts.append(new_kpt[m.trainIdx].pt)
                if m.distance < 0.6*n.distance:
                    best_pts.append(new_kpt[m.trainIdx].pt)

            found = (len(pts) != 0)
            if found:
                pts = np.array(pts, dtype=np.float32)
                rect = cv2.boundingRect(pts)

            if len(best_pts) != 0:
                center_kpts = np.mean(best_pts, axis=0)
            elif len(pts) != 0:
                center_kpts = np.mean(pts, axis=0)

        return found, rect, center_kpts

    def _filterKeypoints(self, keypoints, descriptors):
        new_dsc = []
        new_kpt = []
        inside = self.bbox.inside(keypoints)
        keypoints = np.array(keypoints)
        for descriptor in np.asarray(descriptors[inside]): new_dsc.append(descriptor)
        for keypoint in np.asarray(keypoints[inside]): new_kpt.append(keypoint)
        new_dsc = np.asarray(new_dsc)
        new_kpt = np.asarray(new_kpt)
        return new_kpt, new_dsc

        
class kalman(object):
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 4, 0)

        # self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        cv2.setIdentity(self.kf.transitionMatrix)

        self.kf.measurementMatrix = np.zeros((4, 6), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1
        self.kf.measurementMatrix[1, 1] = 1
        self.kf.measurementMatrix[2, 4] = 1
        self.kf.measurementMatrix[3, 5] = 1

        # self.kf.processNoiseCov = 1e-2 * np.eye(6, dtype=np.float32)
        cv2.setIdentity(self.kf.processNoiseCov, 1e-2)
        self.kf.processNoiseCov[2, 2] = 5
        self.kf.processNoiseCov[3, 3] = 5

        # self.kf.measurementNoiseCov = 1e-1 * np.eye(6, dtype=np.float32)
        cv2.setIdentity(self.kf.measurementNoiseCov, 1e-1)

        # self.kf.errorCovPost = 1. * np.ones((2, 2))
        # self.kf.statePost = 0.1 * np.random.randn(2, 1)
        # print(self.kf.measurementMatrix.shape)
        # print(self.kf.errorCovPre.shape)

    def deltaTime(self, dT):
        self.kf.transitionMatrix[0, 2] = dT
        self.kf.transitionMatrix[1, 3] = dT

    def reset(self, measure):
        self.kf.errorCovPre[0, 0] = 1
        self.kf.errorCovPre[1, 1] = 1
        self.kf.errorCovPre[2, 2] = 1
        self.kf.errorCovPre[3, 3] = 1
        self.kf.errorCovPre[4, 4] = 1
        self.kf.errorCovPre[5, 5] = 1

        state = np.zeros(6, dtype=np.float32)
        state[0] = measure[0]
        state[1] = measure[1]
        state[4] = measure[2]
        state[5] = measure[3]

        self.kf.statePost = state






    
