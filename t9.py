import numpy
import pytesseract
from PIL import Image
import pickle
import xlwt

#图片预处理
def pretreatment(ima):
    ima=ima.convert('L')        #转化为灰度图像
    im=numpy.array(ima)         #转化为二维数组
    for i in range(im.shape[0]):#转化为二值矩阵
        for j in range(im.shape[1]):
            if im[i,j]==75 or im[i,j]==76:
                im[i,j]=1
            else:
                im[i,j]=0
    return im

#提取图片特征
def feature(A):
    midx=int(A.shape[1]/2)+1
    midy=int(A.shape[0]/2)+1
    A1=A[0:midy,0:midx].mean()
    A2=A[midy:A.shape[0],0:midx].mean()
    A3=A[0:midy,midx:A.shape[1]].mean()
    A4=A[midy:A.shape[0],midx:A.shape[1]].mean()
    A5=A.mean()
    AF=[A1,A2,A3,A4,A5]
    return AF

#切割图片并返回每个子图片特征
def incise(im):
    #竖直切割并返回切割的坐标
    a=[];b=[]
    if any(im[:,0]==1):#避免截图没截好的情况
        a.append(0)
    for i in range(im.shape[1]-1):
        if all(im[:,i]==0) and any(im[:,i+1]==1):
            a.append(i+1)
        elif any(im[:,i]==1) and all(im[:,i+1]==0):
            b.append(i+1)
    if any(im[:,im.shape[1]-1]==1):
        b.append(im.shape[1])
    #水平切割并返回分割图片特征
    names=locals()          #初始化分割后子图片的动态变量名
    AF=[]                   #初始化子图片的特征列表
    for i in range(len(a)):
        names['na%s' % i]=im[:,range(a[i],b[i])]
        if any(names['na%s' % i][0,:]==1):
            c=0
        elif any(names['na%s' % i][names['na%s' % i].shape[0]-1,:]==1):
            d=names['na%s' % i].shape[0]-1    
        for j in range(names['na%s' % i].shape[0]-1):
            if all(names['na%s' % i][j,:]==0) and any(names['na%s' % i][j+1,:]==1):
                c=j+1
            elif any(names['na%s' % i][j,:]==1) and all(names['na%s' % i][j+1,:]==0):
                d=j+1
        names['na%s' % i]=names['na%s' % i][range(c,d),:]
        AF.append(feature(names['na%s' % i]))
        for j in names['na%s' % i]:
            print(j)
    return AF

#训练已知图片的特征         
def training():
    train_set={}
    for i in range(11):
        value=[]
        for j in range(15):
            ima=PIL.Image.open('e://bwtest2//'+str(i)+'//'+str(i)+'-'+str(j)+'.png')
            im=pretreatment(ima)
            AF=incise(im)         #切割并提取特征
            value.append(AF[0])
        train_set[i]=value
    #把训练结果存为永久文件，以备下次使用
    output=open('e://bwtest2//train_set.pkl','wb')
    pickle.dump(train_set,output)
    output.close()
    return train_set
    
#计算两向量的距离
def distance(v1,v2):
    vector1=numpy.array(v1)
    vector2=numpy.array(v2) 
    Vector=(vector1-vector2)**2
    distance = Vector.sum()**0.5
    return distance

#用最近邻算法识别单个数字
def knn(train_set,V,k):
    key_sort=[11]*k
    value_sort=[11]*k
    for key in range(11):
        for value in train_set[key]:
            d=distance(V,value)
            for i in range(k):#从小到大排序
                if d<value_sort[i]:
                    for j in range(k-2,i-1,-1):
                        key_sort[j+1]=key_sort[j]
                        value_sort[j+1]=value_sort[j]
                    key_sort[i]=key
                    value_sort[i]=d
                    break
    max_key_count=-1
    key_set=set(key_sort)
    for key in key_set:   #统计每个类别出现的次数
        if max_key_count<key_sort.count(key):
            max_key_count=key_sort.count(key)
            max_key=key
    return max_key

#合并一副图片的所有数字
def identification(train_set,AF,k):
    result=''
    for i in AF:
        key=knn(train_set,i,k)
        if key==10:
            key='.'
        result=result+str(key)
    return float(result)

#识别文件夹中的所有图片上的数据并输出到EXCELL中
def main(k,trained=None,backup=None):
    # k：knn算法的的k，即取训练集中最相邻前k个样本进行判别
    #trained：trained=0则程序会开始训练出训练集，否则会用已保存的训练集
    #backup：backup=0则程序令昨日数据均为0，并备份今日数据到昨日数据和昨日测试数据
    #        backup=1则程序会令昨日数据为昨日测试数据，并备份今日数据到昨日测试数据
    #        backup等于其他时则程序会令昨日数据为昨日数据，并备份今日数据到昨日数据
    if trained==0:
        train_set=training()
    else:
        pkl_file=open('e://bwtest2//train_set.pkl','rb')
        train_set=pickle.load(pkl_file)
        pkl_file.close()
    if backup==0:
        yestoday_data=[0]*32
    elif backup==1:
        pkl_file=open('e://bwtest2//yestoday_data_test.pkl','rb')
        yestoday_data=pickle.load(pkl_file)
        pkl_file.close()
    else:
        pkl_file=open('e://bwtest2//yestoday_data.pkl','rb')
        yestoday_data=pickle.load(pkl_file)
        pkl_file.close()
    backups=[]
    workbook = xlwt.Workbook('e://bwtest2//bwtest2.xls')
    sheet = workbook.add_sheet("record and contrast")
    #设置EXCEL字体为绿色
    font = xlwt.Font()   
    font.colour_index = 3
    style = xlwt.XFStyle()
    style.font = font
    #识别所有图片的数字并输出到EXCELL中
    for i in range(1,33):
        ima=PIL.Image.open('e://bwtest2//test//'+str(i)+'.png')
        im=pretreatment(ima)
        AF=incise(im)
        result=identification(train_set,AF,k)
        backups.append(result)
        sheet.write(i-1, 0, yestoday_data[i-1])
        if result==yestoday_data[i-1]:
            sheet.write(i-1, 1, result,style)
            sheet.write(i-1, 2, '正常')
        else:
            sheet.write(i-1, 1, result)
            sheet.write(i-1, 2, '待定')
    workbook.save('e://bwtest2//bwtest2.xls')
    if backup==0:
        output=open('e://bwtest2//yestoday_data_test.pkl','wb')
        pickle.dump(backups,output)
        output.close()
        output=open('e://bwtest2//yestoday_data.pkl','wb')
        pickle.dump(backups,output)
        output.close()
    elif backup==1:#/yestoday_data_test用来测试之用
        output=open('e://bwtest2//yestoday_data_test.pkl','wb')
        pickle.dump(backups,output)
        output.close()
    else:
        output=open('e://bwtest2//yestoday_data.pkl','wb')
        pickle.dump(backups,output)
        output.close()

main(4,0,0)  #表示：4近邻，还没有训练集备份，还没有昨日数据备份

