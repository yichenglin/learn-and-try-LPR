#基于轮廓的车牌字符分割
#  首先对车牌图像进行预处理，转换为二值图像并除去铆钉
#  然后对图像取轮廓，对轮廓取最小外接矩形，选取大小、位置适合的矩形即为字符
#  对于中文字符，有时要单独处理

import cv2
import numpy as np

__JumpNum=9
__charHigh=15
__charWidth=3
__maxLeft=10
__maxRight=130
__LeftMove=1.15
__charNum=6#除中文字符外还有6个字符

def setJumpNum(num):
    global __JumpNum
    __JumpNum = num

def setcharHigh(high):
    global __charHigh
    __charHigh= high

def setcharWidth(width):
    global __charWidth
    __charWidth=width

def setmaxLeft(left):
    global __maxLeft
    __maxLeft = left

def setmaxRight(right):
    global __maxRight
    __maxRight = right

def setLeftMove(move):
    global __LeftMove
    __LeftMove = move


# 灰度化
# 将图片转换为灰度图,为接下来的二值化作准备
def ImageGrey(src):
    grey = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    return grey


# 二值化
# 将灰度图片进行二值化处理,以便对字符进行提取
# 此处二值化用的是opencv的Otsu'二值化
# 对蓝色车牌识别采用的参数为：cv2.THRESH_OTSU +cv2.THRESH_BINARY
# 对黄色车牌识别用参数cv2.THRESH_BINARY_INV代替cv2.THRESH_BINARY
# 当黑色的方块数目小于一半时，判断为黄色车牌
def ImageThreshold(grey):
    ret, dst = cv2.threshold(grey, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    rows, cols = src_grey.shape
    x = 0
    i = 0
    for i in range(rows):
        j = 0
        for j in range(cols):
            if dst[i, j] == 0:
                x += 1
    if x / (rows * cols) < 0.5:
        ret, dst = cv2.threshold(src_grey, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    return ret, dst


# 去铆钉
# 车牌铆钉与字符连接在一起，会影响字符的分割
# 因此应该对铆钉进行去除
# 我们对二值化的车牌进行横向遍历，每有一次颜色的变化则跳变数加1
# 当每行的跳变数小于某个值时，我们可以认为这行为铆钉，从而将这行全涂黑(即赋值为0)
# 然后按行遍历,去除所有不符合条件的行
# 此处跳变数最小值9是经验权值
def RemoveRivet(dst):
    i = 0
    rows, cols = dst.shape
    for i in range(rows):
        jumpcount = 0
        j = 0
        for j in range(cols - 1):
            if dst[i, j] != dst[i, j + 1]:
                jumpcount += 1
        if jumpcount <= __JumpNum:
            j = 0
            for j in range(cols):
                dst[i, j] = 0
    return dst


# 取轮廓
# 用findContours函数对图像进行取轮廓
# 然后用drawContours函数画出轮廓
def GetContours(src, dst):
    Contours, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_NONE)
    return contours


# 求出每个轮廓的最小外接矩形
# (x,y)为矩形左上角坐标，(w,h)是矩形的宽和高
# 根据x和h的范围，选取大小合适的矩形即为字符
# 此处x的左右范围和字符的高度和宽带是经验权值
# 然后我们将每个选中的矩形的[x,y,w,h]加入一个列表中
# 对列表进行排序，得到按x大小排列的以[x,y,w,h]为元素的列表
def GetRectangle(contours):
    img_ROI = []
    for contour in contours:
        cnt = contour
        x, y, w, h = cv2.boundingRect(cnt)
        if x > __maxLeft:
            if x < __maxRight:
                if h > __charHigh:
                    if w<__charWidth:
                      img_ROI.append([x, y, w, h])

    img_ROI.sort()
    return img_ROI


# 求中文字符
# 因为有的中文有偏旁部首，所以在取轮廓时被截断
# 然后在取矩形时由于大小不符被丢掉
# 对于这种情况，我们对取到的第一个矩形向左移动来截取汉字
# 左移的距离__LeftMove是w的1.15倍，是经验权值
# 先判断列表的元素个数，如果为6个则需要上述操作
def GetChinesecharacter(img_ROI):
    if len(img_ROI) == __charNum:
        img_ROI.insert(0, [int(img_ROI[0][0] - img_ROI[0][2] * __LeftMove),
                           img_ROI[0][1], img_ROI[0][2], img_ROI[0][3]])
    return img_ROI


def charsSegment(src):
    grey = ImageGrey(src)
    ret, dst = ImageThreshold(grey)
    dst = RemoveRivet(dst)
    contours = GetContours(src, dst)
    img_ROI = GetRectangle(contours)
    img_ROI = GetChinesecharacter(img_ROI)
    chars=[]
    for j in range(len(img_ROI)):
        char = dst[img_ROI[j][1]:(img_ROI[j][1] + img_ROI[j][3]),
                  img_ROI[j][0]:(img_ROI[j][0] + img_ROI[j][2])]
        char = cv2.resize(char, (20, 40), interpolation=cv2.INTER_CUBIC)
        chars.append(char)
     #  cv2.imwrite("C:/Users/27161/Desktop/test/character" + str(j + 1) + ".jpg", img_roi)
    return chars


