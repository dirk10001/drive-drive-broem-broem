import cv2


recording = False
cap = None
image_index = 0
video_index = 0

def start():
    global recording
    global video_index
    global cap
    global image_index
    recording = True
    cap = cv2.VideoCapture('car_' + str(video_index) + '.avi')
    image_index = 0
    video_index += 1

def save_image(angle, left, right):
    global image_index
    global recording
    global cap
    if recording and cap.isOpened():
        _, frame = cap.read()
        cv2.imwrite("%s_%03d_%03d_%03d_%03d.png" % (video_file, image_index, angle, left, right), frame)
        image_index += 1


def stop():
    global recording
    cap.release()
    recording = False
