import cv2

recording = False
image_index = 0
video_index = 0
video_file = None

def start():
    global recording
    global image_index
    global video_file
    global video_index
    recording = True
    image_index = 0
    video_file = "run" + str(video_index)
    video_index += 1

def save_image(cap, angle, left, right):
    global image_index
    global video_file
    global recording
    if recording and cap.isOpened():
        _, frame = cap.read()
        cv2.imwrite("%s_%03d_%03d_%03d_%03d.png" % (video_file, image_index, angle, left, right), frame)
        print("write image: " + ("%s_%03d_%03d_%03d_%03d.png" % (video_file, image_index, angle, left, right)))
        image_index += 1


def stop():
    global recording
    recording = False
