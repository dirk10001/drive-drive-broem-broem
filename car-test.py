
import car
import time

car.init()
car.start()
for angle in range(0, 180, 10):
    car.steeringAngle = float(angle)
    car.update()
    time.sleep(1)

car.stop()
car.cleanup()
