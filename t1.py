#!/usr/local/bin/python3
# -*-coding:utf-8 -*-
import cv2
import os
import time

__author__ = "DMOONS"

print("OpenCV Version:", cv2.__version__)

# 保存截图
save_path = './img/'

# 定义摄像头对象，其参数0表示第一个摄像头
camera = cv2.VideoCapture(0)

# 判断视频是否打开
if (camera.isOpened()):
    print('摄像头打开成功!')
else:
    print('摄像头未打开!')

# 测试用,查看视频size
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('Picture size:' + repr(size))

# 帧率
fps = 5
# 总是取前一帧做为背景（不用考虑环境影响）
pre_frame = None

while(1):
    start = time.time()
    # 读取视频流
    ret, frame = camera.read()

    # 转灰度图
    gray_lwpCV = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not ret:
        break
    end = time.time()

    cv2.imshow("capture", frame)

    # 运动检测部分
    seconds = end - start
    if seconds < 1.0 / fps:
        time.sleep(1.0 / fps - seconds)

    gray_lwpCV = cv2.resize(gray_lwpCV, (500, 500))

    # 用高斯滤波进行模糊处理
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

    # 如果没有背景图像就将当前帧当作背景图片
    if pre_frame is None:
        pre_frame = gray_lwpCV
    else:
        # absdiff把两幅图的差的绝对值输出到另一幅图上面来
        img_delta = cv2.absdiff(pre_frame, gray_lwpCV)
        #threshold阈值函数(原图像应该是灰度图,对像素值进行分类的阈值,当像素值高于（有时是小于）阈值时应该被赋予的新的像素值,阈值方法)
        thresh = cv2.threshold(img_delta, 25, 255, cv2.THRESH_BINARY)[1]
        # 膨胀图像
        thresh = cv2.dilate(thresh, None, iterations=2)
        # findContours检测物体轮廓(寻找轮廓的图像,轮廓的检索模式,轮廓的近似办法)
        image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            # 设置敏感度
            # contourArea计算轮廓面积
            if cv2.contourArea(c) < 1000:
                continue
            else:
                print("出现目标物，请求核实")
                # 保存图像
                cv2.imwrite(save_path + str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) + '.jpg', frame)
                break
        pre_frame = gray_lwpCV

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# release()释放摄像头
camera.release()
#destroyAllWindows()关闭所有图像窗口
cv2.destroyAllWindows()
