import bluetooth
import RPi.GPIO as GPIO
from time import sleep


GPIO.setmode(GPIO.BCM)

Motor1a = 19
Motor1b = 6
Motor2a = 13
Motor2b = 5

GPIO.setup(33, GPIO.OUT)
GPIO.setup(34, GPIO.OUT)
GPIO.setup(35, GPIO.OUT)
GPIO.setup(37, GPIO.OUT)




server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
server_sock.bind(("", bluetooth.PORT_ANY))
server_sock.listen(1)

port = server_sock.getsockname()[1]

#uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"

uuid="fe6e06bc-ac1e-11ea-bb37-0242ac130002"

bluetooth.advertise_service(server_sock, "SampleServer", service_id=uuid,
                            service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
                            profiles=[bluetooth.SERIAL_PORT_PROFILE],
                            # protocols=[bluetooth.OBEX_UUID]
                            )

print("Waiting for connection on RFCOMM channel", port)

client_sock, client_info = server_sock.accept()
print("Accepted connection from", client_info)
try:
    while True:
        data = client_sock.recv(1024)
        if not data:
            break
        (x,y,z,s,q) = data
        if y > 47 :
            GPIO.output(Motor1a,GPIO.HIGH)
            GPIO.output(Motor1b,GPIO.LOW)
            GPIO.output(Motor2a,GPIO.LOW)
            GPIO.output(Motor2b,GPIO.HIGH)
        else :
            GPIO.output(Motor1a,GPIO.LOW)
            GPIO.output(Motor1b,GPIO.LOW)
            GPIO.output(Motor2a,GPIO.LOW)
            GPIO.output(Motor2b,GPIO.LOW)
        if z > 124 :
            GPIO.output(Motor1a,GPIO.HIGH)
            GPIO.output(Motor1b,GPIO.LOW)
            GPIO.output(Motor2a,GPIO.LOW)
            GPIO.output(Motor2b,GPIO.LOW)
        else z < 62 :
            GPIO.output(Motor1a,GPIO.LOW)
            GPIO.output(Motor1b,GPIO.LOW)
            GPIO.output(Motor2a,GPIO.HIGH)
            GPIO.output(Motor2b,GPIO.LOW)
        print("Received", data)
except OSError:
    pass

print("Disconnected.")

client_sock.close()
server_sock.close()

print("All done.")
print("BIG DADDY GONNA TAKE A COLD ONE REAL QUICK")
