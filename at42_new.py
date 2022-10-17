import numpy as np
import time
import tensorflow as tf
from robotPi import robotPi
import cv2
from rev_cam import rev_cam  # 摄像头倒转添加
import os
import time
from robotpi_movement import Movement
from robotpi_Cmd import UPComBotCommand

# 1:[1,0,0,0] 前
# 2:[0,1,0,0] 左
# 3:[0,0,1,0] 右
# 4:[0,0,0,1] 后
yuzhi = 110#110 er zhi hua
forwardspeed = 20#20
left_threshold = 1.8e+07
# ssumyuzhi is the ending yuzhi  (way:put thr robot at the ending)
ssumyuzhi = 2.0e+07
# the  time  of  arriving ending
xianzhitime = 36#36
move_times = 2
forwardspeedend = 20
fw_time = 1000#110
pfkernel = np.ones((10, 10), np.uint8)

robot = robotPi()

width = 480
height = 180
channel = 1
inference_path = tf.Graph()
filepath = os.getcwd() + '/model/472/-472'
# 104 is model name


filepath2 = os.getcwd() + '/model/472/-472'

flag = 0
temp_image = np.zeros(width * height * channel, 'uint8')
cap = cv2.VideoCapture(0)
#tiao jie ji ba dong zuo zuo you wei zhi
def adjust():
    print("adjusting")
    mv = Movement()
    while True:
        if cap.isOpened():
            ret, frame = cap.read()
            frame = rev_cam(frame)  # 摄像头倒转添加
            resized_height = int(width * 0.75)
            # 计算缩放比例
            frame = cv2.resize(frame, (width, resized_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            _, frame = cv2.threshold(frame, yuzhi, 255, cv2.THRESH_BINARY)
            # slice the lower part of a frame
            res = frame[resized_height - height:, :]

            cv2.waitKey(1)

            cv2.imshow("frame", res)
            frame = np.array(res, dtype=np.float32)
            ssum = frame.sum()
            print(ssum)
            if ssum > left_threshold:
                print("------------moving left--------------")
                mv.move_left(speed=15)
                # time.sleep(0.2)
            else:
                # pass
                break

#    def moveRight():
#        for i in range(move_times):
#            mv.move_right(speed=15)
#            time.sleep(0.5)

    # print("------------moving right-------------")
    # moveRight()
    # mv.move_right()
    time.sleep(2)
    robot.movement.move_forward(speed=forwardspeedend, times=fw_time)
    time.sleep(1)


def auto_pilot():
    # a = np.array(frame, dtype=np.float32)
    # _, prediction = model.predict(a.reshape(1, width * height))
    # cap = cv2.VideoCapture(0)

    with tf.Session(graph=inference_path) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver1 = tf.train.import_meta_graph(filepath + '.meta')
        #saver2 = tf.train.import_meta_graph(filepath2 + '.meta')
        saver1.restore(sess, filepath)

        tf_X = sess.graph.get_tensor_by_name('input:0')
        pred = sess.graph.get_operation_by_name('pred')
        number = pred.outputs[0]
        prediction = tf.argmax(number, 1)

        time_start = time.time()
        flag = 0
        while cap.isOpened():
            ret, frame = cap.read()
            frame = rev_cam(frame)  # 摄像头倒转添加
            resized_height = int(width * 0.75)
            # 计算缩放比例
            frame = cv2.resize(frame, (width, resized_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            _, frame = cv2.threshold(frame, yuzhi, 255, cv2.THRESH_BINARY)
            # slice the lower part of a frame
            res = frame[resized_height - height:, :]

            cv2.waitKey(1)

            time_end = time.time()
            time_c = time_end - time_start
            # opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, pfkernel)
            if time_c < xianzhitime:
                res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, pfkernel)
            cv2.imshow("frame", res)
            frame = np.array(res, dtype=np.float32)
            print(frame.sum())

            # if time_c<10:
            # forwardspeed=15
            # else:
            # forwardspeed=25
            ssum = frame.sum()
            # if ssum>2.16e+07:
            if ssum > 2.0e+07:
                robot.movement.move_forward(speed=forwardspeed, times=350)
                continue
            if frame.sum() < 700000:
                robot.movement.move_backward(speed=20, times=500)#kan jian hei se hou tui
                time.sleep(0.2)
                robot.movement.turn_right(speed = 10, times = 500)
                #robot.movement.right_ward()
                # robot.movement.right_ward()
                # continue
            # if time_c >= 10 and flag==0:
            # print("dsdssdsdsdsdsds")
            # print()
            # print()
            # saver2.restore(sess, filepath2)

            # tf_X = sess.graph.get_tensor_by_name('input:0')
            # pred = sess.graph.get_operation_by_name('pred')
            # number = pred.outputs[0]
            # prediction = tf.argmax(number, 1)
            # flag=1

            value = prediction.eval(feed_dict={tf_X: np.reshape(frame, [-1, height, width, channel])})
            print('img_out:', value)

            if value == 0:
                print("forward")
                robot.movement.move_forward(speed=forwardspeed, times=200)
            elif value == 1:
                print("left")
                robot.movement.turn_left(speed=15,times=200)
                time.sleep(0.1)
                #robot.movement.left_ward()
            elif value == 2:
                print("right")
                robot.movement.turn_right(speed=12,times=200)
                time.sleep(0.1)
                #robot.movement.right_ward()
            elif value == 3:
                print("stop sign")

                if time_c >= xianzhitime and ssum < ssumyuzhi:
                    adjust()
                    time.sleep(0.5)
                    robot.movement.move_forward(speed=forwardspeedend, times=fw_time)
                    time.sleep(1)
                    robot.movement.hit()
                    break
                else:
                    robot.movement.move_forward(speed=20, times=350)                                   
            elif value == 4:
                print("Banner forward")
                robot.movement.move_forward(speed=20, times=350)
                if time_c >= xianzhitime and ssum < ssumyuzhi:
                    adjust()
                    time.sleep(0.5)
                    robot.movement.move_forward(speed=20, times=fw_time)
                    time.sleep(1)
                    robot.movement.hit()
                    break
            elif value == 5:
                print("Banner left")
                robot.movement.turn_left(speed=12,times=200)
                #robot.movement.left_ward()
                if time_c >= xianzhitime and ssum < ssumyuzhi:
                    adjust()
                    #time.sleep(0.5)
                    robot.movement.move_forward(speed=20, times=fw_time)
                    #time.sleep(1)
                    robot.movement.hit()
                    break
            elif value == 6:
                print("Banner right")
                robot.movement.turn_right(speed=12,times=200)
                #robot.movement.right_ward()
                if time_c >= xianzhitime and ssum < ssumyuzhi:
                    adjust()
                    time.sleep(0.5)
                    robot.movement.move_forward(speed=20, times=fw_time)
                    time.sleep(1)
                    robot.movement.hit()
                    break
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ###############################################################
    # startTime=datetime.datetime.now()
    ###############################################################
    
#    while 1 :
#        speed = 10
#        robot.movement.turn_right(speed=50, time500)
        #robot.movement.turn_right(speed=50, times=500)
        #robot.movement.move_right(speed=50, times=500)
        #robot.movement.left_ward()
    
    
    
   auto_pilot()
    
    #adjust()
    # robot.movement.hit()
    # time.sleep(0.5)
    ##############################################################
    # endTime=datetime.datetime.now()
    # print(endTime-startTime)
    ###############################################################




