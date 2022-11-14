# -*- coding:utf-8 -*-
"""
Author：Huang YuWei
Date：2022/10/15
"""
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
# 定义眼睛的比的阈值
EYE_AR_THRESH = 0.25
# 定义眼睛闪烁阈值，连续闭眼次数
EYE_AR_CONS_FRAMES = 40
# 定义嘴巴的纵横比的阈值
MOU_AR_THRESH = 0.85
# 定义闭眼时间
EYE_CL_TIME=2000

# 计算垂直眼睛界标之间的距离与水平眼睛界标之间的距离之比
def eye_aspect_ratio(eye):
    # 计算垂直线之间的欧氏距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 计算水平面之间的欧氏距离
    C = dist.euclidean(eye[0], eye[3])
    # 计算眼睛比例
    ear = (A + B) / (2.0 * C)
    return ear


# 计算上下嘴唇纵横比
def mouth_aspect_ratio(mou):
    # 计算水平面之间的欧氏距离
    X = dist.euclidean(mou[0], mou[6])
    # 计算垂直线之间的欧氏距离
    Y1 = dist.euclidean(mou[2], mou[10])
    Y2 = dist.euclidean(mou[4], mou[8])
    # 取平均值
    Y = (Y1 + Y2) / 2.0
    # 计算比例
    mar = Y / X
    return mar


# 定义中文输出
def cv2_add_chinese_text(img, text, position, _text_color=(0, 255, 0), _text_size=25):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    _font_style = ImageFont.truetype(
        "simsun.ttc", _text_size, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, _text_color, font=_font_style)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# 判断眼睛，嘴巴
def eyes_mouth_detection(ear, _mouth_ear, frame, _count, yawns, prev_yawn_status,EYE_STATUS,t1):
    # 通过检查眼睛的纵横比，如果眼睛纵横比小于阈值，则为闭眼
    t2 = 0
    if ear < EYE_AR_THRESH:
        if  EYE_STATUS:
            t1=int(round(time.time() * 1000))
        EYE_STATUS = False
        if t1 != 0:
            t2 = int(round(time.time() * 1000))
            print("闭眼时长", int(t2 - t1))
        if (t2 - t1) > EYE_CL_TIME:
            frame = cv2_add_chinese_text(frame, "检测到长时间闭眼！", (120, 5), (255, 0, 0))
        _count += 1
        frame = cv2_add_chinese_text(frame, "闭眼!", (10, 5), (255, 0, 0))
        # 如果闭眼次数大于阈值
        if _count >= EYE_AR_CONS_FRAMES:
            frame = cv2_add_chinese_text(frame, "检测到有睡意，醒醒！", (120, 35), (255, 0, 0))
    # 否则为睁眼
    else:
        EYE_STATUS = True
        _count = 0
        frame = cv2_add_chinese_text(frame, "睁眼", (10, 5), (0, 255, 0))

    # 通过检查嘴唇的纵横比，如果嘴唇纵横比大于阈值，则为打哈欠
    if _mouth_ear > MOU_AR_THRESH:
        frame = cv2_add_chinese_text(frame, "打哈欠!", (10, 35), (255, 0, 0))
        _yawn_status = True
        output_text = "打哈欠次数: " + str(yawns + 1)
        frame = cv2_add_chinese_text(frame, output_text, (10, 70), (0, 0, 225))
    else:
        _yawn_status = False
    if prev_yawn_status == True and _yawn_status == False:
        yawns += 1
    return frame, _count, yawns, _yawn_status,EYE_STATUS,t1


# 检测入口
def driver_fatigue_detection():
    EYE_STATUS = True
    t1=0
    # 加载摄像头，0表示使用默认摄像头，直接写视频路径表示读取该视频
    camera = cv2.VideoCapture("vi.mp4")
    #camera = cv2.VideoCapture(0)
    # 这里直接使用dlib库训练好的人脸68点位特征模型
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    # 初始化基于HOG的dlib人脸识别器
    detector = dlib.get_frontal_face_detector()
    # 加载模型
    predictor = dlib.shape_predictor(predictor_path)

    # 获取左眼特征
    (_lt_start, _lt_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # 获取右眼特征
    (_rt_start, _rt_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # 获取嘴巴特征
    (_mh_start, _mh_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    # 定义已闭眼次数
    COUNTER = 0
    # 定义打哈欠状态
    _yawn_status = False
    # 定义打哈欠次数
    yawns = 0
    # 循环视频
    while True:
        # 读取视频
        ret, frame = camera.read()
        # 定义视频尺寸
        frame = imutils.resize(frame, width=750)
        # 将图像转成灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_yawn_status = _yawn_status
        # 检测灰度框中的人脸
        rects = detector(gray, 0)
        # 循环人脸
        for rect in rects:
            # 确定面部区域的面部特征
            # 将面部坐标（x，y）坐标转换为NumPy
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # 提取左眼和右眼坐标
            _left_eye = shape[_lt_start:_lt_end]
            _right_eye = shape[_rt_start:_rt_end]
            # 提取嘴唇坐标
            mouth = shape[_mh_start:_mh_end]
            # 然后计算双眼的眼睛纵横比的坐标
            _LEFT_EAR = eye_aspect_ratio(_left_eye)
            _RIGHT_EAR = eye_aspect_ratio(_right_eye)
            # 计算上下嘴唇纵横比的坐标
            _MOUTH_EAR = mouth_aspect_ratio(mouth)
            # 计算平均双眼的眼睛纵横比
            _EYE_EAR = (_LEFT_EAR + _RIGHT_EAR) / 2.0

            # 计算双眼和嘴唇的凸包
            _left_eye_hull = cv2.convexHull(_left_eye)
            _right_eye_hull = cv2.convexHull(_right_eye)
            _mouth_hull = cv2.convexHull(mouth)

            # 绘制轮廓
            cv2.drawContours(frame, [_left_eye_hull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [_right_eye_hull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [_mouth_hull], -1, (0, 255, 0), 1)

            # 实时输出眼睛和嘴巴纵横比
            frame = cv2_add_chinese_text(frame, "(眼睛)EAR: {:.2f}".format(_EYE_EAR), (550, 5), (255, 0, 0))
            frame = cv2_add_chinese_text(frame, "(嘴巴)MAR: {:.2f}".format(_MOUTH_EAR), (550, 45), (255, 0, 0))

            # 调用方法进行对比纵横比
            frame, COUNTER, yawns, _yawn_status,EYE_STATUS,t1 = eyes_mouth_detection(_EYE_EAR, _MOUTH_EAR, frame, COUNTER, yawns,
                                                                       prev_yawn_status,EYE_STATUS,t1)

        # 显示视频
        cv2.imshow("video", frame)
        key = cv2.waitKey(1) & 0xFF
        # 按q退出
        if key == ord("q"):
            break
    # 关闭窗口
    cv2.destroyAllWindows()
    # 释放相机流
    camera.release()

if __name__ == '__main__':
    driver_fatigue_detection()
