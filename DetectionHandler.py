from GlobalFileImport import GlobalImport
import numpy as np
import cv2
from FaceDetection import FaceDetection
class DetectHandler:
    def __init__(self):
        self.global_import = GlobalImport()
        if self.global_import.CheckCorrectPath() is None:
            print('error')
            return None

        self.faceDetection = FaceDetection(self.global_import.SHAPE_PREDIRTOR_68_FACE_LANDMARK)

    def Camerahandler(self):
        if self.global_import.CheckCorrectPath() is None:
            print('error')
            return None
        cap = cv2.VideoCapture(0)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if frame is None:
                return
            # Our operations on the frame come here
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Display the resulting frame
            frame =self.faceDetection.Handler(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


def main():
     detectHandler = DetectHandler()
     if detectHandler is None:
         return None
     detectHandler.Camerahandler()

if __name__ == '__main__':
    main()
    