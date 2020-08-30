#!/bin/bash
cd /home/pi
sudo hciconfig hci0 piscan
gpio mode 21 out
gpio mode 22 out
gpio mode 23 out
gpio mode 24 out
v4l2-ctl --set-ctrl=vertical_flip=1
sudo python3 bleucar.py
