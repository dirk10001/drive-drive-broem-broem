import time
import threading
import queue

import bluetooth
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2

import car
import car_ai
import car_capture

# initialize video capture
cap = cv2.VideoCapture(0)

#define car states
STOPPED = 0 # the car is stopped
DRIVING = 1 # the car is being steered with the remote control
AI = 2 # the car is being steered by the neural network
LD = 3 # the car is being steered by the lane driving algorithm

#set the initial state that the the car is stopped
state = STOPPED

#initialize the system as running
running = True

#initialize the car
car.init()

#the driving thread, loop while running and take action based on the current state
def driving(qRead):
    global state
    global running
    while running :
        s = state
        if s == DRIVING:
            #start the car if it is stopped
            if car.stopped :
                car.start()
            try:
                #read the steering angle from the remote control and update the car
                angle = qRead.get(True, 0.2)
                car.steeringAngle = angle
                car.update()
            except queue.Empty as e:
                print('Timeout on get angle from queue')
        elif s == AI:
            #start the car if it is stopped
            if car.stopped :
                car.start()
            #use the neural network to compute the steering angle from the current video frame
            angle = car_ai.compute_steering_angle(cap)
            print("angle " + str(angle))
            car.steeringAngle = angle
            car.update()
        elif s == LD:
            #start the car if it is stopped
            if car.stopped :
                car.start()
            #use the lane algorithm to compute the steering angle from the current video frame
            angle = car_ai.computeSteeringAngleWithHough(cap)
            print("angle " + str(angle))
            car.steeringAngle = angle
            car.update()
        else :
            if not car.stopped :
                #stop the car
                print('STOP CAR')
                car.stop()

#recording thread, capture an image each 200 milliseconds
def recording():
    global running
    global cap
    #if the program is running
    if running :
        #save the current frame
        car_capture.save_image(cap, car.steeringAngle, car.left,car.right)
        #excute the next capture in 200 milliseconds
        threading.Timer(0.2, recording).start()

#controller thread, listen to the bluetooth controller for commands
def controller(qWrite):

    global state
    global running

    #while the program is running
    while running:

        try:
            #create a bluetooth connection
            server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            server_sock.bind(("", bluetooth.PORT_ANY))
            server_sock.listen(1)

            port = server_sock.getsockname()[1]
            uuid="fe6e06bc-ac1e-11ea-bb37-0242ac130002"
            bluetooth.advertise_service(server_sock, "SampleService", service_id=uuid,
                                        service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
                                        profiles=[bluetooth.SERIAL_PORT_PROFILE],
                                        # protocols=[bluetooth.OBEX_UUID]
                                        )
            #wait for the controller to connect
            print("Waiting for connection on RFCOMM channel", port)
            client_sock, client_info = server_sock.accept()
            print("Accepted connection from", client_info)

            try:
                #while there is a connection
                while True:
                    #read data from the controller
                    data = client_sock.recv(1024)
                    if not data:
                        break
                    data = data.decode('utf8')
                    print("Received", data)

                    if data == 'START':
                        print("STARTING")
                        #send ack to the controller
                        client_sock.send(b'\x01')
                        state = DRIVING
                    elif data == 'STOP!':
                        print("STOPPING")
                        #send ack to the controller
                        client_sock.send(b'\x01')
                        state = STOPPED
                    elif data == 'USEAI':
                        print("SWITCH TO AI")
                        #send ack to the controller
                        client_sock.send(b'\x01')
                        state = AI
                    elif data == 'USELD':
                        print("SWITCH TO LANE DETECTION")
                        #send ack to the controller
                        client_sock.send(b'\x01')
                        state = LD
                    elif data == 'S_REC':
                        print("Start recording")
                        #send ack to the controller
                        client_sock.send(b'\x01')
                        #start saving images
                        car_capture.start()
                    elif data == 'Q_REC':
                        print("Stop recording")
                        #send ack to the controller
                        client_sock.send(b'\x01')
                        #stop saving images
                        car_capture.stop()
                    else :

                        if state == DRIVING:
                            try:
                                #if the current state is driving then the data is the current angle.
                                print("STATE " + str(state) + " data " + data)
                                #send the angle to the driving thread
                                qWrite.put(float(data))
                            except Exception as e:
                                print("Not a float " + data)
            except OSError:
                pass

            print("Disconnected.")
            client_sock.close()
            server_sock.close()
            print("Sockets closed")
        except Exception as e:
            #Unexpected error quit running
            print(e)
            running = False

    print("Disconnected.")
    print("All done.")


#create a queue to send angles from the controller thread to the driving thread
q = queue.Queue()
#create driving thread
driving_thread = threading.Thread(target = driving, args = (q,))
#create driving thread
controller_thread = threading.Thread(target = controller, args = (q,))

#start driving thread
driving_thread.start()
#start controller thread
controller_thread.start()
#start recording (schedule a capture each 200 ms, an mage is only saved if the recording is active)
recording()

#wait for the contrller thread to finish
controller_thread.join()
driving_thread.join()
car.cleanup()
