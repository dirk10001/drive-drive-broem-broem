import car_ai
import car_ai_func
import cv2
import os
import fnmatch

data_dir = 'temp'
file_list = os.listdir(data_dir)
image_paths = []
steering_angles = []
pattern = "*.png"
for filename in file_list:
    if fnmatch.fnmatch(filename, pattern):
        image_paths.append(os.path.join(data_dir,filename))
        angle = int(filename[-15:-12])
        steering_angles.append(angle)
        frame = cv2.imread(data_dir + '/' + filename)
        computedAngle = car_ai.compute_steering_angle_from_frame(frame)
        houghAngle = car_ai_func.computeSteeringAngleWithHough(frame)
        print(filename + " " + str(angle) + " " + str(computedAngle) + " " + str(houghAngle))
