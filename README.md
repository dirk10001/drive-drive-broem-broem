
# drive-drive-broem-broem

## Create a trainingset from recorded images
Create a temp directory and copy all recorded images in that directory
```
mkdir temp
cd temp
scp 'pi@192.168.2.30:*.png' .
cd ..
```

Train the model but in order to convert the model the PATH needs to be set:
Linux: export PATH=$PATH:~/.local/bin/
Mac: export PATH=$PATH:~/Library/Python/<version#>/bin


```
python3 learning-model.py
python3 convert-to-tflite.py
```

Copy the trained model to the Pi
```
scp model.tflite pi@192.168.2.30:
```

Login to the Pi and restart the service
```
ssh pi@192.168.2.30
pi@raspberrypi:~ $ sudo systemctl restart broem
```


## Copy files from and to the raspberry pi

Copy all png files from the pi user directory to the current directory
```
scp 'pi@192.168.2.30:*.png' .
```

Copy all python files from the current directory to the pi user directory
```
scp *.py pi@192.168.2.30:
```

## Add a systemd service to run at boottime on the pi
start launch.sh from systemd at boottime
Add system file to: /etc/systemd/system/broem.service
```
Unit]
 Description=Broem Broeme Service
 After=multi-user.target

 [Service]
 Type=idle
 ExecStart=/home/pi/launch.sh

 [Install]
 WantedBy=multi-user.target
 ```

## enable bluetooth
To enable bluetooth first run
```
sudo hciconfig hci0 piscan
```

## Scan for bluetooth devices
```
hcitool scan
```

## Make the raspberry discoverable
```
sudo hciconfig hci0 piscan
```

## Configure bluetooth
```
$ bluetoothctl
[bluetooth]# help
...
devices (list devices)
pairable on
discoverable on
connect <dev>
```
