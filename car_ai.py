import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import car_ai_func

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()


def img_preprocess(image):
    height, _, _ = image.shape
    # image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relevant for lane following
    image = car_ai_func.preProcess(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66)) # input image size (200,66) Nvidia model
    image = image / 255 # normalizing
    return image

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  interpreter.set_tensor(tensor_index, image)

def compute_steering_angle(cap):
    ret, frame = cap.read()
    return compute_steering_angle_from_frame(frame)

def compute_steering_angle_from_frame(frame):
    global interpreter
    output_details = interpreter.get_output_details()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    preprocessed = img_preprocess(frame)
    X = np.array([preprocessed], dtype=np.float32)
    set_input_tensor(interpreter, X)
    interpreter.invoke()
    steering_angle = interpreter.get_tensor(output_details[0]['index'])

    edge = car_ai_func.edgeDetect(frame)

    return round(steering_angle[0][0])


def computeSteeringAngleWithHough(cap):
    ret, frame = cap.read()
    return car_ai_func.computeSteeringAngleWithHough(frame)
