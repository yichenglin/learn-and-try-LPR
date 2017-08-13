# 基于灰度的形态学的车牌定位：
#   首先根据车牌区域中丰富的纹理特征，提取车牌图像中垂直方向的边缘并二值化。
#   然后对得到的二值图像进行数学形态学(膨胀、腐烛、幵闭运算等)的运算，使得车牌区域形成一个闭合的连通区域。
#   最后通过车牌的几何特征(高、宽、宽高比等)对得到的候选区域进行筛选，最终得到车牌图像。

import numpy as np
import cv2

__GaussianKernelSize = 5
__closingKernelSize = np.ones((3, 17), np.uint8)
__SobelOperatorSize = 3

ASPECT_RATIO = 44 / 14  # 中国车牌标准高宽比
AREA = 44 * 14  # 中国车牌标准面积
__maxRatio, __minRatio = ASPECT_RATIO * 1.7, ASPECT_RATIO * 0.8
__maxArea, __minArea = AREA * 20, AREA * 5
__maxAngle = 25


def setGaussianKernelSize(size):
    global __GaussianKernelSize
    __GaussianKernelSize = size


def setClosingKernel(height, width):
    global __closingKernelSize
    __closingKernelSize = np.ones((height, width), np.uint8)


def setSobelOperatorSize(size):
    global __SobelOperatorSize
    __SobelOperatorSize = size


def setMinRatio(x):
    global __minRatio
    __minRatio = ASPECT_RATIO * x


def setMaxRatio(x):
    global __maxRatio
    __maxRatio = ASPECT_RATIO * x


def setMinArea(x):
    global __minArea
    __minArea = AREA * x


def setMaxArea(x):
    global __maxArea
    __maxArea = AREA * x


def setMaxAngle(angle):
    global __maxAngle
    __maxAngle = angle


# 1.预处理：高斯模糊以减少噪点，灰度化以减少处理难度
def __preprocess(img):
    blur = cv2.GaussianBlur(img, (__GaussianKernelSize, __GaussianKernelSize), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    return gray


# 2.提取边缘：
#   用Sobel算子求X方向导数得竖直方向上边缘，并用Otsu（大津算法）二值化
def __extractEdge(img):
    edge = cv2.Sobel(img, cv2.CV_64F, 1, 0, __SobelOperatorSize)
    edge_abs = cv2.convertScaleAbs(edge)
    ret, binary = cv2.threshold(edge_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# 3.数学形态学运算：使得车牌区域形成一个闭合的连通区域
def __morphologicalOperate(img):
    close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, __closingKernelSize)
    return close


# 4.筛选：
#   求连通区域的最小外接矩形，并从中筛选出符合车牌形状的矩形，最后将候选区域调整成统一形状以输出
def __screen(ori, binary):
    image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        # 返回的是Box2D结构：包含矩形某角点的坐标(x, y)，矩形的宽和高(w, h)，以及旋转角度
        if __verify(rect):
            plate = __adjust(ori, rect)
            rois.append(plate)
    return rois


# 调整函数：将符合条件的矩形区域进行旋转、缩放等，输出为统一尺寸
def __adjust(ori, rect):
    points = cv2.boxPoints(rect).astype(int).reshape(-1, 1, 2)

    for point in points:  # 在points中找到左上角点坐标
        if point[0][0] < rect[0][0] and point[0][1] < rect[0][1]:
            x, y = point[0][0], point[0][1]

    width, height, angle = int(rect[1][0]), int(rect[1][1]), rect[2]
    if width < height:
        width, height = height, width
        angle += 90
    rows, cols = ori.shape[0], ori.shape[1]

    rotmat = cv2.getRotationMatrix2D((x, y), angle, 1)
    rotated = cv2.warpAffine(ori, rotmat, (cols, rows))
    plate = cv2.resize(rotated[y:y + height, x:x + width], (136, 36))  # cv2和numpy索引方式不同
    return plate


# 验证函数：通过验证尺寸、高宽比和角度等判断矩形是否符合车牌形状
def __verify(rect):
    global __maxRatio, __minRatio, __maxArea, __minArea, __maxAngle

    area = rect[1][0] * rect[1][1]
    if area < __minArea or area > __maxArea:  return False

    ratio = rect[1][0] / rect[1][1]
    angle = abs(rect[2])
    if rect[1][0] < rect[1][1]:
        ratio = rect[1][1] / rect[1][0]
        angle = 90 - angle

    if __minRatio < ratio < __maxRatio and angle < __maxAngle:
        return True
    return False


def plate_locate(img_address):
    ori = cv2.imread(img_address)
    gray = __preprocess(ori)
    binary = __extractEdge(gray)
    close = __morphologicalOperate(binary)
    rois = __screen(ori, close)

    cv2.imshow("ori", ori)
    cv2.imshow("edge", binary)
    cv2.imshow("close", close)
    for i in range(len(rois)):
        cv2.imshow(str(i), rois[i])
    cv2.waitKey(0)

    #     new_address = os.path.join(img_address, "plates",
    #                                img_address.split(".")[0] + "-" + str(i) + ".jpg")
    #     cv2.imwrite(plate, new_address)


if __name__ == "__main__":
    plate_locate(r"C:\Projects\PR\tests\1.jpg")
