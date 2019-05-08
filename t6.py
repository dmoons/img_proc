#!/usr/bin/env python3
# -*-coding:utf-8 -*-

import time
import picamera

with picamera.PiCamera() as camera:
    camera.resolution = (1024, 768)
    camera.start_preview()
    # 摄像头预热
    time.sleep(2)
    camera.capture('foo.jpg', resize=(320, 240))
    camera.stop_preview()