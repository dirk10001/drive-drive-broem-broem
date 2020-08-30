
import RPi.GPIO as GPIO

LEFTPIN = 6
LEFTPIN_REVERSE = 19

RIGHTPIN = 5
RIGHTPIN_REVERSE = 13

FREQ = 50
PWMLEFT = None
PWMRIGHT = None

steeringAngle = 90.0
left = 100
right = 100
stopped = True

def init():
    global PWMLEFT
    global PWMRIGHT

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LEFTPIN, GPIO.OUT)
    GPIO.setup(RIGHTPIN, GPIO.OUT)
    PWMLEFT = GPIO.PWM(LEFTPIN, FREQ)
    PWMRIGHT = GPIO.PWM(RIGHTPIN, FREQ)

def update():
    global stopped
    global steeringAngle
    global left
    global right
    if not stopped :
        print('update car ' + str(steeringAngle))
        angle = steeringAngle
        if angle < 0 or angle > 180 :
            steeringAngle = 90
            left = 100
            right = 100
        elif angle > 85 and  angle < 95 :
            left = 100
            right = 100
        elif angle <= 85 :
            left = (angle/90.0)*100.0
            right = 100
        elif angle >= 95 :
            left = 100
            right = ((180.0 - angle)/90.0)*100.0
        PWMLEFT.ChangeDutyCycle(left)
        PWMRIGHT.ChangeDutyCycle(right)

def start():
    global stopped
    print('start car')
    PWMLEFT.start(100)
    PWMRIGHT.start(100)
    stopped = False

def stop():
    global stopped
    print('stop car')
    PWMLEFT.stop()
    PWMRIGHT.stop()
    stopped = True

def cleanup():
    GPIO.cleanup()
