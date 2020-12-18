 
import os

class GlobalImport:
    def __init__(self):
        
        self.SHAPE_PREDIRTOR_68_FACE_LANDMARK = 'Models/shape_predictor_68_face_landmarks.dat'
        self.SHAPE_PREDIRTOR_MMOD_VERHICLE = "Models/mmod_rear_end_vehicle_detector.dat"
        self.FILE_PATH_FOR_TRAIN = "Models/hands_training.xml"
        self.SHAPE_PREDIRTOR_HAND_LANDMARK = "Models/Hand_9_Landmarks_Detector.dat"
        self.SHAPE_HAND_DETECTOR_PATH = "Models/HandDetector.svm"
        self.SHAPE_PREDIRTOR_PHONE_LANDMARK = "Models/Phone_4_Landmarks_Detector.dat"
        self.SHAPE_PHONE_DETECTOR_PATH = "Models/PhoneDetector.svm"
        self.SHAPE_PREDIRTOR_ROADMARKER_LANDMARK = "Models/RoadMarker_4_Landmarks_Detector.dat"
        self.SHAPE_ROADMARKER_DETECTOR_PATH = "Models/RoadMarkerDetector.svm"
        self.SHAPE_VERHICLE_DETECTOR_PATH = "Models/car_detector.svm"
        self.SHAPE_VERHICLE_SHAPE_DETECTOR_PATH = "Models/car_detector.dat"
        self.FOLDER_PATH_SAVE_IMAGE = "/ImageSave"

    def  CheckCorrectPath(self):
        if not os.path.isfile(self.SHAPE_PREDIRTOR_68_FACE_LANDMARK):
            print ('face landmark model not exit')
            return None
        return True