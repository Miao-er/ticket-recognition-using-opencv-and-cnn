import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#裁剪并摆正
def align_pic(pic):
    gray_pic = cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY) #灰度图
    _,bin_pic = cv2.threshold(gray_pic,60,255,cv2.THRESH_BINARY) #二值图
    bin_pic = cv2.medianBlur(bin_pic,9) #中值滤波去除细线
    #闭运算去除票区域的黑色杂点
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50,50))
    close_pic = cv2.morphologyEx(bin_pic,cv2.MORPH_CLOSE,close_kernel)
    #开运算去除边角的白色突起
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(140,70))
    open_pic = cv2.morphologyEx(close_pic,cv2.MORPH_OPEN,open_kernel)
    #获取白色票区域的轮廓
    contours,_ = cv2.findContours(open_pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #生成最小矩形区域
    rect = cv2.minAreaRect(contours[0])
    #获取矩形区域的四个顶点，按顺时针
    pt1,pt2,pt3,pt4 = box = np.int0(cv2.boxPoints(rect))
    #计算宽和高
    w = np.sqrt(np.sum(np.power(pt1 - pt2,2)))
    h = np.sqrt(np.sum(np.power(pt1 - pt4,2)))
    #默认宽大于高
    if w < h:
        pt1,pt2,pt3,pt4 = pt2,pt3,pt4,pt1
        w,h = h,w
    points = np.float32([pt1,pt2,pt3])
    points_dst = np.float32([[0,0],[w,0],[w,h]])
    #进行仿射变换，做点对点的映射
    matrix = cv2.getAffineTransform(points,points_dst)
    final_pic = cv2.warpAffine(pic,matrix,(int(w),int(h)))
    final_gray = cv2.warpAffine(gray_pic,matrix,(int(w),int(h)))
    final_pic = cv2.resize(final_pic,(1030,640))
    final_gray = cv2.resize(final_gray,(1030,640))
    shapes = final_pic.shape

    #判断是否上下颠倒，即顶部灰度值低于底部灰度值时
    if final_gray[5][shapes[1]//2] < final_gray[shapes[0] - 5][shapes[1]//2]:
        rotateMatrix = cv2.getRotationMatrix2D((shapes[1]/2,shapes[0]/2),180,1)
        final_pic = cv2.warpAffine(final_pic,rotateMatrix,(shapes[1],shapes[0]))

    return final_pic

#获取向坐标轴投影的直方图
def get_text_project(img_text, mode): 
    pos = []
    #向水平投影
    if mode == 0:
        pos = np.zeros([img_text.shape[1]], dtype=np.int64)
        for i in range(img_text.shape[0]):
            for j in range(img_text.shape[1]):
                if img_text[i, j] == 255:
                    pos[j] += 1
    #向竖直投影
    if mode == 1:
        pos = np.zeros([img_text.shape[0]], dtype=np.int64)
        for i in range(img_text.shape[1]):
            for j in range(img_text.shape[0]):
                if img_text[j, i] == 255:
                    pos[j] += 1
    return pos

#设置最小像素数和最小字符宽度，获取各段范围
def get_peek_range(pos, min_tresh, min_range,is_vtop):
    begin = 0
    # end = 0
    peek_range = []
    for i in range(len(pos)):
        if pos[i] > min_tresh and begin == 0:
            begin = i
        elif pos[i] > min_tresh and begin != 0:
            continue
        elif pos[i] < min_tresh and begin != 0:
            end = i
            if (is_vtop and end-begin < min_range * 1.5 and len(peek_range) == 0):
                continue
            if end-begin >= min_range:
                (x, y) = (begin, end)
                peek_range.append((x, y))
                begin = 0
                # end = 0
        elif pos[i] < min_tresh or begin == 0:
            continue
    return np.asarray(peek_range)

#绘制矩形
def draw_rect(pic,x_range,y_range):
    if len(x_range) == 0 or len(y_range) == 0:
        return pic 
        
    len_x = len(x_range)
    y_range[0]-=2
    y_range[1]+=2
    x_range[0,0]-=2
    x_range[len_x-1,1]+=2
    
    pic = cv2.rectangle(pic,(x_range[0][0],y_range[0]),(x_range[len_x-1][1],y_range[1]),(0,0,255))
    for i in range(len_x - 1):
        pic = cv2.line(pic,((x_range[i][1]+x_range[i+1][0])//2,y_range[0]),((x_range[i][1]+x_range[i+1][0])//2,y_range[1]),(0,0,255))

    return pic   
    
#分割成单个字符序列
def crop_char(pic,x_range,y_range):
    len_x = len(x_range)
    y_range[0]-=2
    y_range[1]+=2
    x_range[0,0]-=2
    x_range[len_x-1,1]+=2
    img_list = []
    for i in range(len_x):
        top = y_range[0]
        bottom = y_range[1]
        if i == 0 and i == len_x - 1:
            left = x_range[0][0]
            right = x_range[0][1]
        elif i == 0:
            left = x_range[0][0]
            right = (x_range[0][1]+x_range[1][0])//2
        elif i == len_x - 1:
            left = (x_range[i-1][1]+x_range[i][0])//2
            right = x_range[i][1]
        else:
            left = (x_range[i-1][1]+x_range[i][0])//2
            right = (x_range[i][1]+x_range[i+1][0])//2
        char_pic = np.copy(pic[top:bottom,left:right])
        char_pic[char_pic==255] = 1 #归一化防止训练时参数过大
        img_list.append(char_pic)

    return img_list

#定位顶部数字区域并分割
def locate_top_pic(pic):
    thresh_low = 50
    thresh_high = 150
    gray_pic = cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY) #灰度图

    #双阈值二值化
    _,low_pic = cv2.threshold(gray_pic,thresh_low,255,cv2.THRESH_BINARY)
    _,high_pic = cv2.threshold(gray_pic,thresh_high,255,cv2.THRESH_BINARY_INV)
    top_pic = cv2.bitwise_and(low_pic,high_pic) 

    #定位顶部数字大致区域
    top,bottom,left,right = 20,100,30,340
    top_part_pic = top_pic[top:bottom,left:right]

    # #中值滤波除去汉字轮廓
    top_bin_pic = cv2.medianBlur(top_part_pic,3)

    #根据垂直方向投影直方图，使用波谷分段
    y_pos = get_text_project(top_bin_pic,1)
    y_range = get_peek_range(y_pos, 20, 30, False)
    x_range = []
    char_list = []
    if len(y_range) != 0:
        y_range = y_range[0]
        #根据水平方向投影直方图，使用波谷分段
        x_pos = get_text_project(top_bin_pic[y_range[0]:y_range[1]][:],0)
        x_range = get_peek_range(x_pos,4,15,True)
        
        #返回分割的单个字符序列
        if len(x_range) != 0:
            char_list = crop_char(top_bin_pic,x_range,y_range)
            #加上相对偏移量
            y_range +=top
            x_range +=left

    return [x_range,y_range],char_list

#定位底部数字区域
def locate_bottom_pic(pic):
    gray_pic = cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY) #灰度图
    
    #二值化图像使得底部数字为最后一组向竖直方向的投影波峰
    _,bin_pic = cv2.threshold(gray_pic,40,255,cv2.THRESH_BINARY_INV)

    #定位顶部数字大致区域
    left,right = 30,800
    
    #根据垂直方向投影直方图，使用波谷分段
    y_pos = get_text_project(bin_pic,1)
    y_range = get_peek_range(y_pos, 80, 20,False)
    x_range = []
    char_list = []
    if len(y_range) != 0:
        y_range = y_range[len(y_range) - 1]
    
        #开运算去除横向单像素宽的游丝
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,2))
        bin_pic = cv2.morphologyEx(bin_pic,cv2.MORPH_OPEN,open_kernel)

        #根据水平方向投影直方图，使用波谷分段
        x_pos = get_text_project(bin_pic[y_range[0]:y_range[1],left:right],0)
        x_range = get_peek_range(x_pos,2,10,False)
        if len(x_range) != 0:
            #加上相对偏移量
            x_range = (x_range + left)
            if len(x_range) >= 21:
                x_range = x_range[:21]
            #返回分割的单个字符序列
            char_list = crop_char(bin_pic,x_range,y_range)

    return [x_range,y_range],char_list
    
#generate train dataset
def load_train_set(train_dir):
    train_list = []
    with open(train_dir + "annotation.txt") as list_data:
        lines = list_data.readlines()
        for line in lines:
            train_list.append(line.strip().split(' '))

    letter_list = []
    letter_labels = []
    number_list = []
    number_labels = []
    for i in range(len(train_list)):
        #读取原始图像
        pic = cv2.imread(train_dir + train_list[i][0]) 
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
        #将标记图像保存，这里训练集不需要
        #cv2.imwrite(segments_dir + train_list[i][0],seg_pic)

    #生成训练网络用的字符集和数字集
        #遍历顶部序列
        if len(top_list) == 9:
            top_list = top_list[2:]
        for idx ,img in enumerate(top_list):
            img = cv2.resize(img,(28,28),interpolation=cv2.INTER_NEAREST)
            if idx == 0:
                letter_list.append(img)
                letter_labels.append(train_list[i][2][0])
            else:
                number_list.append(img)
                number_labels.append(train_list[i][2][idx])
        #遍历底部序列
        for idx, img in enumerate(bottom_list):
            img = cv2.resize(img,(28,28),interpolation=cv2.INTER_NEAREST)
            if idx == 14:
                letter_list.append(img)
                letter_labels.append(train_list[i][1][14])
            else:
                number_list.append(img)
                number_labels.append(train_list[i][1][idx])

    #使用训练集数据，生成字符集的标签，均转换成整型数字便于作为索引
    number_labels = [int(label) for label in number_labels]
    letter_labels = [ord(label) - ord('A') for label in letter_labels]

    letter_list = np.array(letter_list)
    letter_labels = np.array(letter_labels)
    number_list = np.array(number_list)
    number_labels = np.array(number_labels)
    
    return letter_list,letter_labels,number_list,number_labels

