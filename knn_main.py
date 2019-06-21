#main.py: 车牌号图像处理并识别
# 主程序,识别车牌号
import cv2
import numpy as np
import sys
import argparse
import os

def reduce_colors(img, n):
    # 图片切片成3列
    Z = img.reshape((-1,3))
    # 转成float32类型,运算快
    Z = np.float32(Z)
    # 使用k均值进行聚类
    # 迭代停止的模式选择:满足精确度+最大迭代次数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K值,分类数
    K = n
    # 重复进行多少次,将返回最好一次结果
    # 紧密度，返回每个点到相应重心的距离的平方和; labels：标志数组;centers：由聚类的中心组成的数组。
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # 图片重新转成uint8数据类型
    center = np.uint8(center)
    res = center[label.flatten()]
    # 返回图片
    res2 = res.reshape((img.shape))
    return res2

# 图片处理函数
def clean_image(img):
    # RGB转灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 缩放图片
    resized_img = cv2.resize(gray_img
        , None
        , fx=4.0
        , fy=4.0
        , interpolation=cv2.INTER_CUBIC)
    # 进行高斯滤波
    resized_img = cv2.GaussianBlur(resized_img,(5,5),0)
    # 存储高斯结果
    cv2.imwrite('temp/licence_plate_gauss.png', resized_img)
    # 直方图均衡化
    equalized_img = cv2.equalizeHist(resized_img)
    # 存储均衡化直方图
    cv2.imwrite('temp/licence_plate_equ.png', equalized_img)
    # 图片经过聚类处理
    reduced = cv2.cvtColor(reduce_colors(cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR), 8), cv2.COLOR_BGR2GRAY)
    cv2.imwrite('temp/licence_plate_red.png', reduced)
    # 限定阈值二值化图片
    ret, bwimg = cv2.threshold(reduced, 64, 255, cv2.THRESH_BINARY)
    # 存储二值图片
    cv2.imwrite('temp/licence_plate_bwimg.png', bwimg) 
    # 定义一个3*3的十字结构
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    # 对图像进行腐蚀
    erode = cv2.erode(bwimg, kernel, iterations = 1)
    # 存储腐蚀图片
    cv2.imwrite('temp/licence_plate_erode.png', erode)
    return erode

# 提取字符函数
def extract_characters(img):
    # 非运算,将数字部分转成白颜色
    bw_image = cv2.bitwise_not(img)
    cv2.imwrite("temp/bw_image.png",bw_image)
    # 查找边缘
    contours = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    # print(contours)
    char_mask = np.zeros_like(img)
    bounding_boxes = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = w * h
        center = (x + w/2, y + h/2)
        # 对于区域在1000~10000之间的才算字符
        if (area > 1800) and (area < 10000):
            x,y,w,h = x-4, y-4, w+8, h+8
            # 存储图片边框信息
            bounding_boxes.append((center, (x,y,w,h)))
            # 绘画矩形,使用灰度255白色,线宽-1表示填满
            cv2.rectangle(char_mask,(x,y),(x+w,y+h),255,-1)
    # 存储矩形区域
    cv2.imwrite('temp/licence_plate_char_mask.png', char_mask)
    # 将原图和提取的字符图进行与与运算提取出字符图,在进行非运算
    clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask = bw_image))
    # 根据字符中心进行从左到右排序
    bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])  
    # 裁出字符图片
    characters = []
    for center, bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = clean[y:y+h,x:x+w]
        characters.append((bbox, char_image))
    # 返回整张字符图和裁剪后字符图
    return clean, characters

# 高亮字符图
def highlight_characters(img, chars, file):
    # 图片转RGB
    # if(file=="CU5600"):
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for bbox, char_img in chars:
        x,y,w,h = bbox
        # 将所选区域用白色绘制
        cv2.rectangle(output_img,(x,y),(x+w,y+h),255,1)
    # 存储高亮字符图片
    cv2.imwrite('temp/licence_plate_out.png', output_img)
    return output_img

# 开始执行
#命令形式： python main.py --image T1.jpg
# 从命令行获取要读取的车牌号图片   
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "输入图片路径")
# args = vars(ap.parse_args())
# # 读取图片
# img = cv2.imread(args["image"])
# 读取字符特征和标签
samples = np.loadtxt('char_samples.data',np.float32)
label = np.loadtxt('char_label.data',np.float32)
label = label.reshape((label.size,1))
# 使用K近邻模型并训练
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, label)
# 重定向输出流
orig_stdout = sys.stdout
f = open('output.txt','w')
sys.stdout = f
path = "data" #文件夹目录  
files= os.listdir(path)
total=0
for file in files:
    # 读取图片
    img = cv2.imread(path+"/"+file)
    # print(file)
    img = cv2.resize(img,(108, 24))
    # 处理图片
    img = clean_image(img)
    # 提取图片中字符
    clean_img, chars = extract_characters(img)
    # 高亮图片中字符
    output_img = highlight_characters(clean_img, chars,file[0:-4])
    plate_chars = ""
    for bbox, char_img in chars:
        # 将提取的char转成字符特征形式
        # print("char_img.shape")
        # print(char_img.shape)
        small_img = cv2.resize(char_img,(10,10))
        small_img = small_img.reshape((1,100))
        small_img = np.float32(small_img)
        retval, results, neigh_resp, dists = model.findNearest(small_img, k = 1)
        plate_chars += str(chr((results[0][0])))
    # 存储识别的车牌号
    print("识别车牌号: %s" % plate_chars)
    print("实际车牌号: %s" % file[0:-4])
    print()
    if(plate_chars==file[0:-4]):
        total = total + 1
sys.stdout = orig_stdout
f.close()
print("准确率", 1.00*total/len(files))
