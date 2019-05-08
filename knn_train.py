#train.py: 提取字符特征，KNN初步测试
# 将字符图片转成特征，并生成各个特征的label
import cv2
import numpy as np
# 生成车牌号中字符 ord由字符生成Ascii码 chr由Ascii码生成字符
CHARS = [chr(ord('0') + i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
# print(CHARS)
# 加载字符图片函数
def load_char_images():
    characters = {}
    for char in CHARS:
        # 以灰度模式读取
        char_img = cv2.imread("chars/%s.png" % char, 0)
        characters[char] = char_img
    return characters

# 加载字符图片
characters = load_char_images()
# 特征长度为100
samples =  np.empty((0,100))
for char in CHARS:
    char_img = characters[char]
    # 将图片缩放成10*10,保持特征长度的一致性
    small_char = cv2.resize(char_img,(10,10))
    # 图片进行切片成1*100，一维向量
    sample = small_char.reshape((1,100))
    # 添加进samples
    samples = np.append(samples,sample,0)
# 生成字符label，用列存储
label = np.array([ord(c) for c in CHARS],np.float32)
label = label.reshape((label.size,1))
# 存储字符特征和字符label
np.savetxt('char_samples.data',samples)
np.savetxt('char_label.data',label)
# 模型训练
# 读取字符特征和标签
samples = np.loadtxt('char_samples.data',np.float32)
label = np.loadtxt('char_label.data',np.float32)
label = label.reshape((label.size,1))
# 使用K近邻进行学习
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, label)
# 读取一个测试样本
# img = cv2.imread("chars/2.png",0)
total=0
for char in CHARS:
    img = cv2.imread("chars/%s.png" %char,0)
    small_img = cv2.resize(img,(10,10))
    ret, small_img = cv2.threshold(small_img, 64, 255, cv2.THRESH_BINARY)
    # print(small_img)
    small_img = small_img.reshape((1,100))
    small_img = np.float32(small_img)
    # 使用K近邻进行预测
    retval, results, neigh_resp, dists = model.findNearest(small_img, k = 1)
    print(chr(results[0][0]))
    if(chr(results[0][0])==char):
        total += 1
print(1.0*total/len(CHARS))