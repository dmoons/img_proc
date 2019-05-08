#!/usr/bin/python
# -*- coding:utf-8 -*-

import picamera
from time import sleep
from fractions import Fraction

with picamera.PiCamera() as camera:
    camera.resolution = (1280, 720)
    # 设置帧率为1/6fps,然后将曝光时间设置为6秒
    # 最后将iso参数设置为800
    camera.framerate = Fraction(1, 6)
    camera.shutter_speed = 6000000
    camera.exposure_mode = 'off'
    camera.iso = 800
    # 给摄像头一个比较长的预热时间，让摄像头尽可能的适应周围的光线
    # 你也可以试试开启AWB【自动白平衡】来代替长时间的预热
    sleep(10)
    # 最后捕捉图像，因为我们将曝光时间设置为6秒，所以拍摄的时间比较长
    camera.capture('/tmp/dark.jpg')