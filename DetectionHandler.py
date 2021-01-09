from operator import le
import os
from GlobalFileImport import GlobalImport
import numpy as np
import cv2
from FaceDetection import FaceDetection
from PhoneDetection import PhoneDetection
from VerhicleDetection import VerhicleDetection
from Roadmarkerdetect import RoadmarkerDetection
import datetime
from time import gmtime, strftime
class DetectHandler:
    def __init__(self):
        self.global_import = GlobalImport()
        if self.global_import.CheckCorrectPath() is None:
            print('error')
            return None

        self.faceDetection = FaceDetection(self.global_import.SHAPE_PREDIRTOR_68_FACE_LANDMARK)
        self.phoneDetatecton = PhoneDetection(self.global_import.SHAPE_PHONE_DETECTOR_PATH,self.global_import.SHAPE_PREDIRTOR_PHONE_LANDMARK)
        self.verhicleDetection = VerhicleDetection(self.global_import.SHAPE_VERHICLE_DETECTOR_PATH, self.global_import.SHAPE_VERHICLE_SHAPE_DETECTOR_PATH)
        self.roadmarkerDetection = RoadmarkerDetection(self.global_import.SHAPE_ROADMARKER_DETECTOR_PATH,self.global_import.SHAPE_PREDIRTOR_ROADMARKER_LANDMARK )
    
    def Camerahandler(self):
        if self.global_import.CheckCorrectPath() is None:
            print('error')
            return None
        cap = cv2.VideoCapture(0)
        count = 0
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if frame is None:
                return
            count += 1
            EyeObject_arr = []
            YawnMouthObject_arr=[]
            looking_other_way_status_arr = []
            dets = []
            if count%6 == 0:

                frame,EyeObject_arr,YawnMouthObject_arr, looking_other_way_status_arr = self.faceDetection.Handler(frame)
            #if count%5 == 0:
                #frame, dets = self.phoneDetatecton.Handler(frame)
            if count%4 == 0:
                frame, dets = self.verhicleDetection.Handler(frame)
            # Our operations on the frame come here
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Display the resulting frame
            cv2.imshow('frame',frame)
            if len(EyeObject_arr) > 0 or len(YawnMouthObject_arr) > 0 or len(looking_other_way_status_arr) > 0 or len(dets) > 0:
               self.SaveImage(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    def SaveImage(self,frame):
        output_folder = 'Output/'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)       
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file_name = str(current_time).replace('-','_')
        file_name = file_name.replace(' ','_')
        file_name = file_name.replace(':','_')
        file_name = file_name.replace('.','_')
        file_name = output_folder + file_name + '.png'
        cv2.imwrite(file_name, frame) 



def main():
     detectHandler = DetectHandler()
     if detectHandler is None:
         return None
     detectHandler.Camerahandler()

if __name__ == '__main__':
    main()
    