import sys
import os
import dlib
import glob
from cv2 import *
import collections
import numpy
from numpy.lib.shape_base import vsplit
from scipy.spatial import distance

EyeObject = collections.namedtuple('EyeObject','leftEar rigthEar averageAspectRatio')
MouthObject = collections.namedtuple('MouthObject','inSideMount outSideMount averageAspectRatio')

class FaceDetection:
    def __init__(self,detectorPath):
        self.detectorPath = detectorPath
        self.resize = 2
        self.two_eyelid_aspect_ratio_standard = 0.4
        self.mouth_aspect_ratio_standard = 0.45

        self.sleeping_status = 'Sleeping'
        self.yawning_mouth_status = "Yawning mouth"

        self.InitialDetector(self.detectorPath)
        
    def InitialDetector(self, detectorPath):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(detectorPath)
        self.win = dlib.image_window()
    
    def Handler(self,frame):
        shape_arr= self.FaceDetection(frame)
        if shape_arr is None:
            print('shape is incorrect')
            return
        frame, EyeObject_arr = self.EyeSleepDetection(frame, shape_arr)
        if frame is None:
            print('frame is empty')
            return
        frame, YawnMouthObject_arr = self.YawnMouthDetection(frame, shape_arr)
        return frame

    def FaceDetection(self, frame):
        if frame is None:
            print('Frame is none/empty')
            return None 
        width = int(frame.shape[1] * self.resize )
        height = int(frame.shape[0] * self.resize)
        dim = (width, height)
        frame = cv2.resize(frame, (width, height),interpolation = cv2.INTER_AREA) 

        dets = self.detector(frame, 1)
        print("Number of faces detected: {}".format(len(dets)))
        shape_arr = []
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
            shape = self.predictor(frame, d)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))

            part_arr = []
            if shape is None:
                return None
            for i in range(shape.num_parts):
                shape.part(i).x = shape.part(i).x * self.resize
                shape.part(i).y = shape.part(i).y * self.resize
                part_arr.append(shape.part(i))
            shape_arr.append(part_arr)
        return shape_arr

    def EyeSleepDetection(self, frame, shape_arr):
            #                   left eye                     right eye
        #=|P2-P6|          P38  |  P39                    P44  |  P45
        #=|p3-P5|      P37 -----|----- P40     nose   P43 -----|----- P46
        #=|p1-p4|          P42  |  P41                    P48  |  P47
        if frame is None:
            print('Frame is empty')
            return 
        
        EyeObject_arr = []

        for vec in shape_arr:
            
            ratioPoint37To39 = distance.euclidean((vec[36].x,vec[36].y), (vec[39].x,vec[39].y))#cv2.norm(vec[36]-vec[39],cv2.NORM_L2); #=|P37-P39|
            ratioPoint38To42 = distance.euclidean((vec[37].x,vec[37].y), (vec[41].x,vec[41].y)) #=|P38-P42|
            ratioPoint39To41 = distance.euclidean((vec[38].x,vec[38].y), (vec[40].x,vec[40].y)) #=|P39-P41|
            #ratioPoint39To41 = cv2.norm(cv2.Mat(vec[38]), cv2.Mat(vec[40])); #=|P39-P41|
            #compute Left EAR
            leftEar= (ratioPoint38To42 + ratioPoint39To41) / (2.0 * ratioPoint37To39);
            #compute right EAR
            ratioPoint43To46 = distance.euclidean((vec[42].x,vec[42].y), (vec[45].x,vec[45].y))#=|P43-P46|
            ratioPoint44To48 = distance.euclidean((vec[43].x,vec[43].y), (vec[47].x,vec[47].y)) #=|P44-P48|
            ratioPoint45To47 = distance.euclidean((vec[44].x,vec[44].y), (vec[46].x,vec[46].y))#=|P45-P47|
            rigthEar = (ratioPoint44To48 + ratioPoint45To47) / (2.0 * ratioPoint43To46);

            averageAspectRatio=(leftEar+rigthEar)/2;
            EyeObject_arr.append(EyeObject(leftEar, rigthEar, averageAspectRatio))

        # detecting slepping status
        sleeping_status_arr = []
        for eyeObject in EyeObject_arr:
            if eyeObject.averageAspectRatio < self.two_eyelid_aspect_ratio_standard:
                sleeping_status_arr.append(self.sleeping_status)

        frame = self.DrawEye(frame, shape_arr)
        number_of_sleep_status = 'Number of ' + self.sleeping_status + str(len(sleeping_status_arr))
        cv2.putText(frame,number_of_sleep_status, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

        return frame, EyeObject_arr
    #def CaculateDistance(self, point1,point2):

    def DrawEye(self, frame, shape_arr):
        scalar = (0,255,0);
        for part in shape_arr:
            # Left eye
            for i in range(37,42):
                # draw with opencv
                frame= cv2.line(frame, (part[i].x,part[i].y), (part[i-1].x,part[i-1].y), scalar,1)
            frame = cv2.line(frame, (part[36].x,part[36].y), (part[41].x,part[41].y),scalar,1)

            # Right eye
            for i in range(43,48):
                frame = cv2.line(frame, (part[i].x,part[i].y), (part[i-1].x,part[i-1].y),scalar,1)

            frame = cv2.line(frame, (part[42].x,part[42].y), (part[42].x,part[42].y),scalar, 1)
        return frame;

    def YawnMouthDetection(self, frame, shape_arr):
                #              P51             P53
        #          P50                       P54
        #              P62     P63     P64
        #     P49 P61                         P65 P55
        #              P68     P67     P66
        #          P60                       P56
        #              P59             P57
        YawnMouthObject_arr = []
        for vec in shape_arr:
            # compute InSideMount
            ratioPoint61To65 = distance.euclidean((vec[60].x,vec[60].y), (vec[64].x,vec[64].y)) #=|P61-P65|
            ratioPoint62To68 = distance.euclidean((vec[61].x,vec[61].y), (vec[67].x,vec[67].y)) #=|P62-P68|
            ratioPoint63To67 = distance.euclidean((vec[62].x,vec[62].y), (vec[66].x,vec[66].y)) #=|P62-P66|
            ratioPoint64To66 = distance.euclidean((vec[63].x,vec[63].y), (vec[65].x,vec[65].y)) #=|P64-P66|

            inSideMount= (ratioPoint63To67+ ratioPoint64To66 ) / (2.0 * ratioPoint61To65);

        # compute OutSideMount
            ratioPoin49To55  = distance.euclidean((vec[48].x,vec[48].y), (vec[54].x,vec[54].y)) #=|P49-P55|
            ratioPoint50To60 = distance.euclidean((vec[49].x,vec[49].y), (vec[59].x,vec[59].y))#=|P50-P60|
            ratioPoint51To59 = distance.euclidean((vec[50].x,vec[50].y), (vec[58].x,vec[58].y)) #=|P51-P59|
            ratioPoint53To57 = distance.euclidean((vec[52].x,vec[52].y), (vec[56].x,vec[56].y)) #=|P52-P57|
            ratioPoint54To56 = distance.euclidean((vec[53].x,vec[53].y), (vec[55].x,vec[55].y)) #=|P54-P56|

            outSideMount= (ratioPoint50To60 + ratioPoint51To59+ ratioPoint53To57 +ratioPoint54To56) / (4.0 * ratioPoin49To55);

            averageAspectRatio= (inSideMount+ outSideMount)/2;
            YawnMouthObject_arr.append(MouthObject(inSideMount, outSideMount, averageAspectRatio))
        
        frame = self.DrawMouth(frame, shape_arr)

         # detecting yawning mouth status
        yawning_mouth_status_arr = []
        for mouthObject in YawnMouthObject_arr:
            if mouthObject.averageAspectRatio > self.mouth_aspect_ratio_standard:
                yawning_mouth_status_arr.append(self.yawning_mouth_status)

        frame = self.DrawEye(frame, shape_arr)
        number_of_sleep_status = 'Number of ' + self.yawning_mouth_status + str(len(yawning_mouth_status_arr))
        cv2.putText(frame,number_of_sleep_status, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

        return frame, YawnMouthObject_arr

    def DrawMouth(self, frame, shape_arr):
        scalar = (0,255,0);
        for part in shape_arr:
            # Left eye
            for i in range(49,60):
                # draw with opencv
                cv2.line(frame, (part[i].x, part[i].y), (part[i-1].x,part[i-1].y), scalar, 1)
            cv2.line(frame, (part[48].x,part[48].y), (part[59].x,part[59].y),scalar, 1)

            # Right eye
            for i in range(61,68):
                cv2.line(frame, (part[i].x,part[i].y), (part[i-1].x,part[i-1].y),scalar, 1)

            cv2.line(frame, (part[60].x,part[60].y), (part[67].x,part[67].y),scalar, 1)
        return frame


    def EarAndNoseDetection(self, frame, shape_arr):
        pass
