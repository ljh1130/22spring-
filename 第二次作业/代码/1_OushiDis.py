# 手写数字特征向量与标准值之间的欧氏距离

import numpy
import matplotlib.pyplot as plt
import math
import cv2


# 注：需要安装opencv
# pip3 install opencv-python
# ----------------------------------
# 读入手写体的图像数据
def read_local_MNIST_dataset(filename=r"/Users/lijihang/PycharmProjects/模式识别机器视觉实验/第二次作业/dataset/mnist.npz"):
    f = numpy.load(filename)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


read_local_MNIST_dataset()


# ----------------------------------
# 读入标准样本的图像数据
def read_std_digit_images(src_dir):
    filenames_list = ["0.png", "1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png"]
    images_list = [None] * len(filenames_list)
    for idx, filename in enumerate(filenames_list):
        pathfilename = src_dir + "/" + filename
        print(pathfilename)
        img_data = cv2.imread(pathfilename, cv2.IMREAD_GRAYSCALE)  # 读取一幅标准图片的数据
        if False:
            cv2.imshow('image%d' % idx, img_data)
            cv2.waitKey(0)
        images_list[idx] = img_data
        # print(img_data.shape)
    return images_list


# ----------------------------------

(X, Y), (A, B) = read_local_MNIST_dataset()

std_digits_imgs = read_std_digit_images(src_dir="/Users/lijihang/PycharmProjects/模式识别机器视觉实验/第二次作业/dataset/std_digits")

N = 10000  # 取N幅图片
M = 4  # 每幅图片压缩为 M 维
GRID_LINES = int(28 / M) + 1  # 这里采用按5格均分割图片矩阵行，每一格所包含的行数
# 这里处理方法是 取阈值为256的一半(128=0x80)
# >阈值则【黑色点数】计数+1 ;
threshold1 = 128
fin = [] #存放结果的列表
# 计算手写体的特征向量矩阵 手写体的数字是 空白为0 黑色为255
features_matrix = numpy.empty(shape=[0, M])  # 特征矩阵初始化为空矩阵
for i in range(N):
    img_matrix = X[i]  # 读入每一个数字
    fin.append(Y[i])
    # print("img_matrix.shape",img_matrix.shape)
    features_vect1 = [[0 for i in range(4)] for i in range(4)]  # 特征向量初始化


    for line, row in enumerate(img_matrix):  # 每行line=0,..,27
        i = 0  # 记录每一行的列数
        for e in row:  # 行里每个像素
            feature_idx_x = line // GRID_LINES  # 整除 得到当前行line应该落在哪一个格子的横坐标中
            feature_idx_y = i // GRID_LINES  # 整除 得到当前列i应该落在哪一个格子的纵坐标中
            i += 1
            if e >= threshold1:  # 二值化:大于阈值 则看成白色 小于则看成黑色 0-黑色 255-白色
                features_vect1[feature_idx_x][feature_idx_y] += 1  # 往下为新增
                # print('0', end="")
            # else:
                # print(' ', end="")
        # print(" ")

    # for line,row in enumerate(img_matrix):

    features_matrix = numpy.append(features_matrix, numpy.array(features_vect1).reshape(M, M), axis=0)
    print("features_matrix:")
    print(features_matrix)

print("手写体特征向量矩阵\n", features_matrix)

threshold2 = 128
# 计算标准数字的特征向量矩阵
# 灰度图像是数字0-黑色 255-白色
std_digits_features_matrix = numpy.empty(shape=[0, M])  # 特征矩阵初始化为空矩阵
for img in std_digits_imgs:
    features_vect2 = [[0 for i in range(4)] for i in range(4)]  # 特征向量初始化
    # print("img.shape",img.shape)
    for line, row in enumerate(img):  # 每行line=0,..,27
        # print(row)
        i = 0
        for e in row:  # 行里每个像素
            # print(e," ")
            feature_idx_x = line // GRID_LINES  # 整除 得到当前行line应该落在哪一个格子(0,1,2,3)中
            feature_idx_y = i // GRID_LINES  # 整除 得到当前列i应该落在哪一个格子的纵坐标中
            i += 1
            if e <= threshold2:  # 二值化:大于阈值 则看成白色 小于则看成黑色 0-黑色 255-白色
                features_vect2[feature_idx_x][feature_idx_y] += 1
    # test = numpy.array(features_vect2).reshape(M,M)
    std_digits_features_matrix = numpy.append(std_digits_features_matrix, numpy.array(features_vect2).reshape(M, M),
                                              axis=0)
    # test1 =  numpy.append(std_digits_features_matrix,numpy.array(features_vect2).reshape(M,M),axis=0)
#
print("标准数字特征向量矩阵\n", std_digits_features_matrix)

print("手写体与各个标准数字特征向量的欧氏距离\n")

# for i,vector in enumerate(features_matrix):
#     for j,std_vect in enumerate(std_digits_features_matrix):
#         data_tup = zip(vector,std_vect)
#         dist =0.0
#         for x,y in data_tup:
#             dist += (x-y)*(x-y)
#         print("distance of(i=%d,j=%d)"%(i,j),vector,std_vect,math.sqrt(dist))
#
i = 0
j = 0
fea_idx = 0 #计数，记录已经将第几行手写数字特征矩阵输入了
std_idx = 0 #计数，记录已经将第几行标准数字特征矩阵输入了
vector = numpy.empty(shape=[0, M]) #存储手写数字数据集
std_vect = numpy.empty(shape=[0, M]) #存储标准数字数据集
min_dist = 999999999.0 #定义欧氏距离最小值
pre_fin = 20 #最终预测的结果
correct_cnt = 0.0 #预测正确的结果

for test_fea in features_matrix:  # 第一层循环，首先将4个行向量提取出来，组合到一个numpy数组中
    if (i == 0):
        vector = numpy.empty(shape=[0, M])
        vector = numpy.append(vector, numpy.array(test_fea).reshape(1, M), axis=0)
        i += 1
        continue
    else:
        vector = numpy.append(vector, numpy.array(test_fea).reshape(1, M), axis=0)
        if i != 3:
            i += 1
            continue
        else:
            i = 0
            fea_idx += 1
            std_idx = 0
            min_dist = 9999999999.0  # 初始化欧氏距离最小值
            pre_fin = 20  # 初始化预测结果
    for test_std in std_digits_features_matrix:
        #首先按照数字序号，读取标准数字数据集
        if (j == 0):
            std_vect = numpy.empty(shape=[0, M])
            std_vect = numpy.append(std_vect, numpy.array(test_std).reshape(1, M), axis=0)
            j += 1
            continue
        else:
            std_vect = numpy.append(std_vect, numpy.array(test_std).reshape(1, M), axis=0)
            if j != 3:
                j += 1
                continue
            else:
                j = 0
                std_idx += 1

        # data_tup = zip(vector,std_vect)
        # dist =0.0
        # for x,y in data_tup:
        #     dist += (x-y)*(x-y)

        dist = 0.0
        mat_trans = numpy.transpose(vector - std_vect) #矩阵转置
        test = numpy.dot(mat_trans, vector - std_vect) #欧氏距离计算公式，向量和转置向量点乘
        dist = math.sqrt(math.fabs(numpy.sum(test))) #矩阵值相加，并开平方
        # dist = numpy.sum(test)
        if(dist < min_dist):
            min_dist = dist
            pre_fin = std_idx
        # print("distance of(i=%d,j=%d)"%(fea_idx,std_idx),vector,std_vect,math.sqrt(dist))
        # print("distance of(i=%d,j=%d)" % (fea_idx, std_idx-1), vector, std_vect, dist)
    # print("该数字预测结果为：%d"%(pre_fin-1))
    if(fin[fea_idx-1] == pre_fin-1):
        correct_cnt+=1

print("正确率为：",float(correct_cnt) / float(N))





