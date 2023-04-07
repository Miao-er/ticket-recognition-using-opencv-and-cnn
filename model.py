import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
from torch.autograd import Variable
import numpy as np
from preprocess import *

#两层（卷积层 + 最大池化层 + relu激活函数）+两层全连接层输出在十个方向与概率正相关的值
class Net(nn.Module):
    def __init__(self,output_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, output_classes)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        
#train
def evaluation(predicted, truth):
    _, class_predicted = torch.max(predicted.data, 1) #返回每一个样本最大值和索引
    return 100.0* (class_predicted == truth).sum()/ predicted.size(0) #计算准确率

def split_data(data,labels):
    #随机生成乱序索引
    index = np.random.permutation(len(data))
    #7：3分割训练集和数据集及其标签
    len_train = int(0.7*len(index))
    train_index = index[0:len_train]
    test_index = index[len_train:]

    train_data = data[train_index]
    train_labels = labels[train_index]
    test_data = data[test_index]
    test_labels = labels[test_index]

    return train_data,train_labels,test_data,test_labels

def train(net,epoch,data,labels,batch_size,optimizer):
    net.train()
    indices = collections.deque()
    indices.extend(np.random.permutation(len(data)))
    iter = 0
    ave_loss = 0
    accu = 0
    while len(indices) >= batch_size: #每次取batch_size个字符进行训练
        batch_idx = [indices.popleft() for i in range(batch_size)] 
        train_x, train_y = data[batch_idx], labels[batch_idx] 
        train_x = Variable(torch.FloatTensor(train_x) , requires_grad=False) 
        train_y = train_y.astype(np.int64) 
        train_y = Variable(torch.LongTensor(train_y), requires_grad=False) 
        output = net(train_x)
        #计算交叉熵损失
        loss = nn.CrossEntropyLoss()(output,train_y)   
        #反向传播，更新参数
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #平均损失值
        ave_loss +=loss
        #准确率
        accu += evaluation(output,train_y.data)
        #batch迭代次数
        iter += 1
    print("train  epoch:{},loss:{},accu:{}".format(epoch,ave_loss/iter,accu/iter))

#训练数据中分割出来的测试集
def test(net,epoch,data,labels,batch_size):
    net.eval()
    indices = collections.deque()
    indices.extend(np.random.permutation(len(data)))
    iter = 0
    ave_loss = 0
    accu = 0
    with torch.no_grad(): #测试时不需要累积梯度
        while len(indices) >= batch_size:
            batch_idx = [indices.popleft() for i in range(batch_size)]
            test_x, test_y = data[batch_idx], labels[batch_idx]
            test_x = Variable(torch.FloatTensor(test_x) , requires_grad=False) 
            test_y = test_y.astype(np.int64) 
            test_y = Variable(torch.LongTensor(test_y), requires_grad=False) 
            output = net.forward(test_x) 
            loss = nn.CrossEntropyLoss()(output,test_y)  
            ave_loss +=loss
            accu += evaluation(output,test_y.data)
            iter += 1
        print("test  epoch:{},loss:{},accu:{}".format(epoch,ave_loss/iter,accu/iter))

#用来预测单个字符的分类
def eval(net,data):
    net.eval()
    with torch.no_grad():
        test_x = Variable(torch.FloatTensor(data) , requires_grad=False) 
        test_x = test_x.unsqueeze(0) #添加维度，使得batch_size = 1

        output = net.forward(test_x) 
    #返回索引值，即标签
    return np.argmax(output[0]).numpy()

#生成已训练的数字识别网络
def get_NUM_net(data,labels,learning_rate,momentum,epoches,test_mode):
    NUM_net = Net(10)
    train_data,train_labels = data,labels
    test_data,test_labels =[],[]
    if not test_mode:
        train_data,train_labels,test_data,test_labels = split_data(data,labels)
    optimizer = optim.SGD(NUM_net.parameters(), lr=learning_rate,momentum=momentum)
    for epoch in range(epoches):
        train(NUM_net,epoch,train_data,train_labels,64,optimizer)
        if not test_mode:
            test(NUM_net,epoch,test_data,test_labels,64)
    return NUM_net

#生成已训练的字母识别网络
def get_LETTER_net(data,labels,learning_rate,momentum,epoches,test_mode):
    LETTER_net = Net(26)
    train_data,train_labels = data,labels
    test_data,test_labels =[],[]
    if not test_mode:
        train_data,train_labels,test_data,test_labels = split_data(data,labels)
    optimizer = optim.SGD(LETTER_net.parameters(), lr=learning_rate,momentum=momentum)
    for epoch in range(epoches):
        train(LETTER_net,epoch,train_data,train_labels,8,optimizer)
        if not test_mode:
            test(LETTER_net,epoch,test_data,test_labels,8)
    return LETTER_net

#无标签数据加载和预测接口
def test_set(test_dir,segments_dir,LETTER_net,NUM_net):
    test_list = []
    with open(test_dir + "annotation.txt") as list_data:
        lines = list_data.readlines()
        for line in lines:
            test_list.append(line.strip().split(" "))

    for i in range(len(test_list)):
        test_list[i] = test_list[i][0]
        #读取原始图像
        pic = cv2.imread(test_dir + test_list[i]) 
        pic = cv2.resize(pic,(1080,1200))
        #生成矫正图像
        rotate_pic = align_pic(pic)
        #创建副本以标记矩形
        seg_pic = np.copy(rotate_pic)
        #定位顶部序列位置，返回相应的分割字符
        top_box,top_list = locate_top_pic(rotate_pic) 
        #定位底部序列位置，返回相应的分割字符
        bottom_box,bottom_list = locate_bottom_pic(rotate_pic)    
        #在图像上标记分割矩形
        seg_pic = draw_rect(seg_pic,*top_box)
        seg_pic = draw_rect(seg_pic,*bottom_box)
        #将标记图像保存
        cv2.imwrite(segments_dir + test_list[i],seg_pic)

        #遍历顶部序列
        top_string = ""
        if len(top_list)  != 0:
            for idx ,img in enumerate(top_list):
                img = cv2.resize(img,(28,28),interpolation=cv2.INTER_NEAREST)
                if idx == 0:
                    output = eval(LETTER_net,img)
                    ch = chr(output + ord('A'))#将标签修改为字母
                    top_string += ch
                else:
                    output = eval(NUM_net,img)
                    top_string += str(output)

        #遍历底部序列
        bottom_string = ""
        if bottom_list != 0:
            for idx, img in enumerate(bottom_list):
                img = cv2.resize(img,(28,28),interpolation=cv2.INTER_NEAREST)
                if idx == 14:
                    output = eval(LETTER_net,img)
                    ch = chr(output + ord('A')) #将标签修改为字母
                    bottom_string += ch
                else:
                    output = eval(NUM_net,img)
                    bottom_string += str(output)
        #写入预测结果
        with open(test_dir + "prediction.txt","a") as f:
            f.write(test_list[i]+" "+bottom_string+" "+top_string+"\n")


