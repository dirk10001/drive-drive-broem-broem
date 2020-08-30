import cv2
import car_capture
import time

cap=cv2.VideoCapture(0)

car_capture.start()
for i in range(0, 4):
    car_capture.save_image(cap,120,80,100)
    time.sleep(1)

car_capture.stop()
