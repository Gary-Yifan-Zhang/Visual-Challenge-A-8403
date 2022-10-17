import numpy as np
import tensorflow as tf
from robotPi import robotPi
import cv2
from rev_cam import rev_cam  # 摄像头倒转添加
import os
import time
from robotpi_movement import Movement

# 作者：19级机器人工程 (按拼音排序) 桂源泽 苏琦 杨一 张亦凡
# 日期：2021年11月13日 20：33
# 作品用途：视觉对抗A项目自动驾驶主程序

# 1:[1,0,0,0] 前
# 2:[0,1,0,0] 左
# 3:[0,0,1,0] 右
 # 4:[0,0,0,1] 后

'''
阈值时间v
多次判断x
飘移x
霍夫圆判断v
socket网络传输（骚）x
信息熵v
高斯滤波x
卡尔曼滤波x
9秒跑完赛道法获得训练集

'''
# 自动驾驶参数：
threshold_yuzhi = 110  # 二值化阈值
forwardspeed = 40  # 前进速度
forwardspeedend = 25  # 最终击靶速度
xianzhitime = 50  # 驶过障碍的时间

# 击靶参数
fw_times = 100  # 击靶前进时间
fw_speed = 20  # 击靶前进速度
ssumyuzhi = 2.0e+07  # 画面数组最大值
flag_max = 2  # 数组判断最大值
MinRadius = 0  # 最小霍夫圆圈
MaxRadius = 300  # 最大霍夫圆圈
minyuanxinyuzhi = 200  # 最小圆心阈值
maxyuanxinyuzhi = 280  # 最大圆心阈值
red_yuzhi = 5  # 红色二值化阈值
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

robot = robotPi()

# 图像处理参数
width = 480
height = 180
channel = 1
temp_image = np.zeros(width * height * channel, 'uint8')
cap = cv2.VideoCapture(0)
#cap1 = cv2.VideoCapture(1)

# tf参数
inference_path = tf.Graph()
filepath = os.getcwd() + '/model/266/-266'
# /number is model name



def adjust():  # 击靶调整
    print("recognising hough circles")

    time_d = 0#调整时间
    adjust_flag = 0#左右旋转参数

    cv2.destroyAllWindows()# 关闭所有窗口，即cap
    cap.release()
    cap1 = cv2.VideoCapture(1)# 调用击靶摄像头
    
    while True:
        if cap1.isOpened():
            time_hough_start = time.time()
            # print(time_hough_start)
            while time_d < 5:
                ret, frame1 = cap1.read()

                # cv2.imshow('res', res1)
                frame1 = rev_cam(frame1)
                resized_height1 = int(width * 0.75)
                frame1 = cv2.resize(frame1, (width, resized_height1))
                cv2.waitKey(1)

                hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)  # HSV空间
                lower_red = np.array([130, 50, 100], np.uint8)  # 设定红色的阈值
                upper_red = np.array([255, 255, 255], np.uint8)
                # cv2.imshow('frame', frame1)
                mask = cv2.inRange(hsv, lower_red, upper_red)  # 设定取值范围
                res1 = cv2.bitwise_and(frame1, frame1, mask=mask)  # 对原图像处理

                # cv2.imshow('res1',res1)
                grey = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
                # cv2.imshow('res', res1)
                _, grey = cv2.threshold(grey, red_yuzhi, 128, cv2.THRESH_BINARY)
                # grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, pfkernel)#画面开运算
                grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel)
                # cv2.imshow('res1',res1)
                circles1 = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 16, 1000, param1=100, param2=10, minRadius=MinRadius,
                                            maxRadius=MaxRadius)

                if np.all(circles1) == 1:
                    time_hough_end = time.time()
                    # print(time_hough_end)
                    circles = circles1[0, :, :]  # 提取为2维
                    circles = np.uint16(np.around(circles))  # 四舍五入，取整
                    for i in circles[:]:
                        cv2.circle(grey, (i[0], i[1]), i[2], (255, 100, 100), 5)  # 画圆
                        cv2.circle(grey, (i[0], i[1]), 2, (255, 100, 100), 10)  # 画圆心
                    for j in circles[:]:
                        yuanxinx = j[0]
                    cv2.line(grey, (minyuanxinyuzhi, 0), (minyuanxinyuzhi, 480), (255, 100, 100), 2, 4)
                    cv2.line(grey, (maxyuanxinyuzhi, 0), (maxyuanxinyuzhi, 480), (255, 100, 100), 2, 4)
                    cv2.waitKey(1)
                    cv2.imshow('res', grey)
                    time_d = time_hough_end - time_hough_start
                    print(time_d)
                    if yuanxinx >= minyuanxinyuzhi and yuanxinx <= maxyuanxinyuzhi:
                        print("forward")
                        robot.movement.move_forward(speed=2, times=10)#前进
                    elif yuanxinx >= maxyuanxinyuzhi:
                        print("turn right")
                        robot.movement.turn_right(speed=3, times=100)#右转
                    else:
                        print("turn left")
                        robot.movement.turn_left(speed=3, times=100)#左转
                        continue
                else:
                    adjust_flag = adjust_flag + 1 # 瞎几把转转找圆 roboty twisty
                    print("no circle")#未识别圆圈
                    if adjust_flag % 2 == 0:
                        robot.movement.turn_left(speed=5, times=100 * adjust_flag)
                        time.sleep(1)
                    else: 
                        robot.movement.turn_right(speed=5, times=100 * adjust_flag)
                        time.sleep(1)
                    continue
            robot.movement.move_forward(speed=fw_speed,  times=fw_times)#前进
            time.sleep(1)
            robot.movement.wave_hands()#击打标靶
            print("hit")
            y = input("stop")#停止
            time.sleep(2000)
        else:
            print("hough error")#摄像头未打开
            continue
    time.sleep(1)


def auto_pilot():  # 自主前进程序
    with tf.Session(graph=inference_path) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.import_meta_graph(filepath + '.meta')  # 调用训练的模型
        saver.restore(sess, filepath)

        tf_X = sess.graph.get_tensor_by_name('input:0')  # 调用所需要的参数
        pred = sess.graph.get_operation_by_name('pred')
        number = pred.outputs[0]
        prediction = tf.argmax(number, 1)

        time_start = time.time()  # 定义开始时间
        
        while True:
            ret, frame = cap.read()  # 读取摄像头画面到 frame和ret
            frame = rev_cam(frame)  # 摄像头倒转
            resized_height = int(width * 0.75)  # 定义画面新高度为480*0.75

            frame = cv2.resize(frame, (width, resized_height))  # 画面分辨率调整
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 画面转化为灰度

            _, frame = cv2.threshold(frame, threshold_yuzhi, 255, cv2.THRESH_BINARY)
            res = frame[resized_height - height:, :]  # 选取摄像头下半部分

            cv2.waitKey(1)

            time_end = time.time()  # 定义当前时间
            time_c = time_end - time_start  # 定义当期运行时间

            # res = cv2.morphologyEx(res, cv2.MORPH_OPEN, pfkernel)#画面开运算
            # res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, pfkernel)#画面闭运算

            cv2.imshow("frame", res)  # 显示res图像画面
            frame = np.array(res, dtype=np.float32)  # 将res数据转化成numpy数组形式
            ssum = frame.sum()  # 定义画面数字总和
            print(ssum)  # 印出画面中所有数字总和

            if time_c < 10:  # 第一个转弯速度慢点
                forwardspeed = 15
            else:
                forwardspeed = 25

            value_true = prediction.eval(feed_dict={tf_X: np.reshape(frame, [-1, height, width, channel])})  # 预测当前画面模型

            # # 进入机器人运动判断：
            # flag = [0, 0, 0, 0, 0, 0, 0]  # 判断次数数组，每一位代表判断的类型
            #
            # while max(flag) < flag_max:  # 每一位最大
            #     # 判断模型：
            #     value = prediction.eval(feed_dict={tf_X: np.reshape(frame, [-1, height, width, channel])})  # 预测当前画面模型
            #     print("img_out:", value)
            #
            #     if value == 0:
            #         print("0+")
            #         flag[0] += 1
            #
            #         print("forward")
            #         robot.movement.move_forward(speed=forwardspeed, times=200)
            #     elif value == 1:
            #         print("1+")
            #         flag[1] += 1
            #     elif value == 2:
            #         print("2+")
            #         flag[2] += 1
            #     elif value == 3:
            #         print("stop sign")
            #         flag[3] += 1
            #     elif value == 4:
            #         print("Banner forward")
            #         flag[4] += 1
            #     elif value == 5:
            #         print("Banner left")
            #         flag[5] += 1
            #     elif value == 6:
            #         print("Banner right")
            #         flag[6] += 1
            #     elif cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            #
            # value_ture = np.argmax(np.array(flag))  # 返回最大数值索引值，即预测最大的位置

            if value_true == 0:
                print("forward")
                robot.movement.move_forward(speed=forwardspeed, times=350)
            elif value_true == 1:
                print("left")
                # robot.movement.drift_left(speed=15, times=70)
                # time.sleep(0.1)
                robot.movement.left_ward()
            elif value_true == 2:
                print("right")
                # robot.movement.drift_right(speed=15, times=70)
                # time.sleep(0.1)
                robot.movement.right_ward()
            elif value_true == 3:
                print("stop sign")
                if time_c >= xianzhitime and ssum < ssumyuzhi:
                    adjust()
                    break
                else:
                    print("forward")
                    robot.movement.move_forward(speed=20, times=350)
            elif value_true == 4:
                print("Banner forward")
                robot.movement.move_forward(speed=20, times=350)
                if time_c >= xianzhitime and ssum < ssumyuzhi:
                    adjust()
                    break
            elif value_true == 5:
                print("Banner left")
                # robot.movement.turn_left(speed=12, times=200)
                # robot.movement.left_ward()
                if time_c >= xianzhitime and ssum < ssumyuzhi:
                    cap.release()
                    adjust()
                    break
            elif value_true == 6:
                print("Banner right")
                # robot.movement.turn_right(speed=12, times=200)
                # robot.movement.right_ward()
                if time_c >= xianzhitime and ssum < ssumyuzhi:
                    adjust()
                    break
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # flag = [0, 0, 0, 0, 0, 0, 0]

        cv2.destroyAllWindows()


if __name__ == '__main__':
    ###############################################################
    # startTime=datetime.datetime.now()
    ###############################################################

    # while 1 :
    #     speed = 10
    #     robot.movement.turn_right(speed=50, time=500)
    # robot.movement.turn_right(speed=50, times=500)
    # robot.movement.move_right(speed=50, times=500)
    # robot.movement.left_ward()

    x = input("################waiting for begin####################")

    auto_pilot()

    # adjust()
    # robot.movement.hit()
    # time.sleep(0.5)
    ##############################################################
    # endTime=datetime.datetime.now()
    # print(endTime-startTime)
    ###############################################################
