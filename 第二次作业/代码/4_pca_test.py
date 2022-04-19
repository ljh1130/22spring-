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

N = 1000  # 取N幅图片
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


def pca(x,k=0,percent = 0.9):
    """
    算法思想：将原始向量特征值大小进行排序，并按大到小顺序输出下标。最后将每一个特征值加和得到总数，将每一个特征值除以总和即为贡献率
    从大到小将每一个特征值相加，加到百分之九十即停止。并把对应的向量保存
    """
    m,n = x.shape
    mean = numpy.mean(x,axis=0)
    mean.shape = (n,1)
    x_norm = x - mean
    x_norm = x_norm.T  # 将它变成 行列分别为特征的矩阵 便于计算！！！
    cov = numpy.dot(x_norm, x_norm.T) #计算协方差矩阵
    eigval, eigvec = numpy.linalg.eig(cov) #计算cov矩阵的特征值和右特征向量
    index = numpy.argsort(-eigval) #输出数组-eigval排序后的索引值，从大到小的顺序
    eigvec_sort = eigvec[index] # 将特征向量按照索引顺序进行排序
    eigval_sort = eigval[index] # 将特征值按照索引值进行排序
    eigval_ratio = eigval_sort/numpy.sum(eigval_sort)
    sum = 0
    for i in range(eigval_ratio.shape[0]):# 将的贡献值从大到小相加，当加和大于等于90时，保留此时全部向量
        sum += eigval_ratio[i]
        if sum > percent:
            return eigvec_sort[:,:i+1]

# eigVec = pca(features_matrix)
# print(eigVec)

def neartemplet_ou(x_train,y_train,sample): #模板匹配算法
    """
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param sample: 样本
    """
    fea_idx = 0
    std_idx = 0
    correct_cnt = 0
    # sample = numpy.dot(sample, eigVec)
    while fea_idx < (N*4):  # 第一层循环，首先将4个行向量提取出来，组合到一个numpy数组中
        vector = x_train[fea_idx:fea_idx+4,:]
        fea_idx += 4
        min_dist = 9999999999.0 #初始化最短距离
        std_idx = 0
        eigVec = pca(vector) #计算每一个数字降维向量
        mean = numpy.mean(vector) #计算vector的均值
        vector = vector - mean #对待测样本去均值
        vector = numpy.dot(vector, eigVec)

        while std_idx < 40:
            std_vect = sample[std_idx:std_idx+4,:]
            std_vect = std_vect - mean # 对标准值去均值
            std_vect = numpy.dot(std_vect,eigVec)
            std_idx += 4

            dist = 0.0
            mat_trans = numpy.transpose(vector - std_vect)  # 矩阵转置
            test = numpy.dot(mat_trans, vector - std_vect)  # 欧氏距离计算公式，向量和转置向量点乘
            dist = math.sqrt(math.fabs(numpy.sum(test)))  # 矩阵值相加，并开平方
            # dist = numpy.sum(test)
            if (dist < min_dist):
                min_dist = dist
                predict_fin = std_idx / 4 - 1 #最终预测结果
        if (fin[int(fea_idx / 4 - 1)] == int(predict_fin)):
            correct_cnt += 1
    return correct_cnt


def neartemplet_ma(x_train,y_train,sample): #模板匹配算法
    """
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param sample: 样本
    """
    fea_idx = 0
    std_idx = 0
    correct_cnt = 0
    # sample = numpy.dot(sample, eigVec)
    while fea_idx < (N*4):  # 第一层循环，首先将4个行向量提取出来，组合到一个numpy数组中
        vector = x_train[fea_idx:fea_idx+4,:]
        fea_idx += 4
        min_dist = 9999999999.0 #初始化最短距离
        std_idx = 0
        eigVec = pca(vector) #计算每一个数字降维向量
        mean = numpy.mean(vector) #计算vector的均值
        vector = vector - mean #对待测样本去均值
        vector = numpy.dot(vector, eigVec)

        while std_idx < 40:
            std_vect = sample[std_idx:std_idx+4,:]
            std_vect = std_vect - mean # 对标准值去均值
            std_vect = numpy.dot(std_vect,eigVec)
            std_idx += 4

            dist = 0.0
            com_vecstd = numpy.append(vector, std_vect, axis=1)  # 将两个矩阵vector和std_vect合起来
            #以下代码是消除零行零列，使矩阵可逆
            idx = numpy.argwhere(numpy.all(com_vecstd[..., :] == 0, axis=0))  # 找到零列
            com_vecstd = numpy.delete(com_vecstd, idx, axis=1)  # 删除零列
            idx1 = numpy.argwhere(numpy.all(com_vecstd[:, ...] == 0, axis=1))  # 找到零行
            com_vecstd = numpy.delete(com_vecstd, idx1, axis=0)  # 删除零行
            mat_trans = com_vecstd.T  # 矩阵转置
            cov = numpy.cov(mat_trans)# 求转置矩阵的协方差矩阵
            # invD = numpy.linalg.inv(cov)
            invD = numpy.linalg.pinv(cov) # 求协方差矩阵的逆矩阵
            dist = numpy.dot(numpy.dot(mat_trans, com_vecstd), invD)
            # 转置后的矩阵和删除零行之后的拼接矩阵进行点乘，将结果和逆矩阵点乘
            # dist = mat_trans * mat_conv * com_vecstd
            dist = numpy.sum(dist)# 将每一维的值相加，求和
            # dist = mahalanobis(vector, std_vect)
            dist = math.fabs(dist)
            if (dist < min_dist):
                min_dist = dist
                predict_fin = std_idx / 4 - 1
        if (fin[int(fea_idx / 4 - 1)] == int(predict_fin)):
            correct_cnt += 1
    return correct_cnt

"""
开始测试：
"""
# eigVec = pca(features_matrix)
# mean = numpy.mean(features_matrix,axis=0).reshape((1,4))
# mean = numpy.mean(features_matrix,axis=0)
#去均值
# x_train = features_matrix - mean
# sample = std_digits_features_matrix - mean
x_train = features_matrix
sample = std_digits_features_matrix
#降维
# x_train = numpy.dot(x_train,eigVec)
# sample =  numpy.dot(sample,eigVec)
#模板匹配
# print(fin)
ans_ou = neartemplet_ou(x_train,fin,sample)
ans_ma = neartemplet_ma(x_train,fin,sample)
print("欧氏距离正确率为：",float(ans_ou) / float(N))
print("马氏距离正确率为：",float(ans_ma) / float(N))