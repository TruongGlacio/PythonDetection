import sys
import os
import dlib
import glob
from cv2 import *
import collections
import numpy
from numpy.lib.function_base import append
from numpy.lib.shape_base import vsplit
from scipy.spatial import distance

class RoadmarkerDetection:
    def __init__(self,detectorPath,shape_detector_Path):
        self.detectorPath = detectorPath #.svm file
        self.shape_detector_Path = shape_detector_Path #.dat file
        self.resize = 1
        self.InitialDetector(self.shape_detector_Path,self.detectorPath)


    def InitialDetector(self, shape_detector_Path, detectorPath):
        print('detectorPath==>',detectorPath)
        self.detector = dlib.simple_object_detector(detectorPath)# deserialize .svm file
        # or self.detector = dlib.fhog_object_detector(detectorPath)
        #self.shape_predictor = dlib.shape_predictor(shape_detector_Path) # deserialize .dat file
        
    def Handler(self, frame):
        frame = self.RoadmarkerDetection(frame)
        return frame

    def RoadmarkerDetection(self, frame):
        if frame is None:
            print('Frame is none/empty')
            return None 
        width = int(frame.shape[1] * self.resize )
        height = int(frame.shape[0] * self.resize)
        dim = (width, height)
        frame = cv2.resize(frame, (width, height),interpolation = cv2.INTER_AREA) 
        
        dets = self.detector(frame, 1)
        scalar = (255,0,0)
        print("Number of road marker detected: {}".format(len(dets)))
        #cv2.putText(frame, "Number of phone detected: {}".format(len(dets)), (p.x + 4, p.y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))
        if len(dets)> 0:    
            cv2.putText(frame,"Road detected: {}".format(len(dets)), (20,170), cv2.FONT_HERSHEY_SIMPLEX,0.5, scalar)
        for d in dets:
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), 255, 1)
            # shape = self.shape_predictor(frame, d)
            # for i in range(shape.num_parts):
            #     p = shape.part(i)
            #     cv2.circle(frame, (p.x, p.y), 2, 255, 1)
            #     cv2.putText(frame, str(i), (p.x + 4, p.y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))
        return frame