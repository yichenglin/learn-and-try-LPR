import cv2
import numpy as np
from matplotlib import pyplot as plt

_ImageLenth = 140
_ImageWidth = 40


# 载入图像
# 并将图像的格式转换为140*40像素
def ImageLoad():
    src = cv2.imread("1.jpg")
    src = cv2.resize(src, (_ImageLenth, _ImageWidth), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("原图像", src)
    cv2.waitKey(0)
    return src


# 灰度化
# 将图片转换为灰度图,为接下来的二值化作准备
def ImageGrey(src):
    src_grey = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    cv2.imshow("灰度化", src_grey)
    cv2.waitKey(0)
    return src_grey


# 二值化
# 将灰度图片进行二值化处理,以便对字符进行提取
# 此处二值化用的是opencv的Otsu'二值化
# 对蓝色车牌识别采用的参数为：cv2.THRESH_OTSU +cv2.THRESH_BINARY
# 对黄色车牌识别用参数cv2.THRESH_BINARY_INV代替cv2.THRESH_BINARY
# 此处缺少对黄色车牌和蓝色车牌的判断，应该改进
def ImageThreshold(src_grey):
    ret, dst = cv2.threshold(src_grey, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    cv2.imshow("二值化图像", dst)
    cv2.waitKey(0)
    return ret, dst


# 去铆钉
# 车牌铆钉与字符连接在一起，会影响字符的分割
# 因此应该对铆钉进行去除
# 我们对二值化的车牌进行横向遍历，每有一次颜色的变化则跳变数加1
# 当每行的跳变数小于某个值时，我们可以认为这行为铆钉，从而将这行全涂黑(即赋值为0)
# 然后按行遍历,去除所有不符合条件的行
# 此处跳变数最小值x=9是经验权值
def RemoveRivet(dst):
    x = 9
    i = 0
    rows, cols = dst.shape
    for i in range(rows):
        jumpcount = 0
        j = 0
        for j in range(cols - 1):
            if dst[i, j] != dst[i, j + 1]:
                jumpcount += 1
        if jumpcount <= x:
            j = 0
            for j in range(cols):
                dst[i, j] = 0
    cv2.imshow("去除铆钉的图像", dst)
    cv2.waitKey(0)
    return dst


# 取轮廓
# 用findContours函数对图像进行取轮廓
# 然后用drawContours函数画出轮廓
def GetContours(src, dst):
    Contours, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_NONE)
    Contours = cv2.drawContours(src, contours, -1, (0, 0, 255), 1)
    cv2.imshow("取轮廓", Contours)
    cv2.waitKey(0)
    return contours


# 求出每个轮廓的最小外接矩形
# (x,y)为矩形左上角坐标，(w,h)是矩形的宽和高
# 根据x和h的范围，选取大小合适的矩形即为字符
# 此处x和h的值为经验权值
# 然后我们将每个选中的矩形的[x,y,w,h]加入一个列表中
# 对列表进行排序，得到按x大小排列的以[x,y,w,h]为元素的列表
def GetRectangle(contours):
    img_ROI = []
    for contour in contours:
        cnt = contour
        x, y, w, h = cv2.boundingRect(cnt)
        if x > 10:
            if x < 130:
                if h > 15:
                    img_ROI.append([x, y, w, h])

    img_ROI.sort()
    return img_ROI


# 求中文字符
# 因为有的中文有偏旁部首，所以在取轮廓时被截断
# 然后在取矩形时由于大小不符被丢掉
# 对于这种情况，我们对取到的第一个矩形向右移动来截取汉字
# 右移的距离是w的1.15倍，是经验权值
# 先判断列表的元素个数，如果为6个则需要上述操作
def GetChinesecharacter(img_ROI):
    if len(img_ROI) == 6:
        s = 1.15
        img_ROI.insert(0, [int(img_ROI[0][0] - img_ROI[0][2] * s),
                           img_ROI[0][1], img_ROI[0][2], img_ROI[0][3]])
    return img_ROI


# 截取图片
# 对上面的所有矩形进行截取，则可以得到单个字符
def InterceptCharacter(src, dst, img_ROI):
    for j in range(len(img_ROI)):
        img = cv2.rectangle(src, (img_ROI[j][0], img_ROI[j][1]),
                            (img_ROI[j][0] + img_ROI[j][2],
                             img_ROI[j][1] + img_ROI[j][3]),
                            (0, 255, 0), 1)
        cv2.imshow("最小外接矩形", img)
        img_roi = dst[img_ROI[j][1]:(img_ROI[j][1] + img_ROI[j][3]),
                  img_ROI[j][0]:(img_ROI[j][0] + img_ROI[j][2])]
        cv2.imshow("字符", img_roi)
        img_roi = cv2.resize(img_roi, (20, 40), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("character" + str(j + 1) + ".jpg", img_roi)
        cv2.waitKey(0)


src = ImageLoad()
src_grey = ImageGrey(src)
ret, dst = ImageThreshold(src_grey)
dst = RemoveRivet(dst)
contours = GetContours(src, dst)
img_ROI = GetRectangle(contours)
img_ROI = GetChinesecharacter(img_ROI)
InterceptCharacter(src, dst, img_ROI)
cv2.destroyAllWindows()
