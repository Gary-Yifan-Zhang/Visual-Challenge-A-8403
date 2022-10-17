from rev_cam import rev_cam
import numpy as np
import time
import cv2

MinRadius = 30
MaxRadius = 100
minyuanxinyuzhi = 180
maxyuanxinyuzhi = 300
width = 480
height = 180
channel = 1
red_yuzhi = 5  # 二值化阈值

temp_image = np.zeros(width * height * channel, 'uint8')
cap1 = cv2.VideoCapture(1)
yuanxinx = 0
#pfkernel = np.ones((10, 10), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))


def adjust():#
    print("recognising hough circles")
    while True:
        if cap1.isOpened():
            ret, frame1 = cap1.read()

            #cv2.imshow('res', res1)
            frame1 = rev_cam(frame1)
            resized_height1 = int(width * 0.75)
            frame1 = cv2.resize(frame1, (width, resized_height1))
            cv2.waitKey(1)
            hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)  # HSV空间
            lower_red = np.array([130,50,100],np.uint8)  # 设定红色的阈值
            upper_red = np.array([255,255,255],np.uint8)
            #cv2.imshow('frame', frame1)
            mask = cv2.inRange(hsv, lower_red, upper_red)  # 设定取值范围
            res1 = cv2.bitwise_and(frame1, frame1, mask=mask)  # 对原图像处理
            
            #cv2.imshow('res1',res1)
            grey = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('res', res1)
            _,grey = cv2.threshold(grey, red_yuzhi, 128, cv2.THRESH_BINARY)
            #grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, pfkernel)#画面开运算
            grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel)
            #cv2.imshow('res1',res1)
            circles1 = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT , 16, 1000, param1 = 100 ,param2 = 10, minRadius=MinRadius , maxRadius=MaxRadius)
            if np.all(circles1) == 1:
                circles = circles1[0, :, :]  # 提取为2维
                circles = np.uint16(np.around(circles))  # 四舍五入，取整
                for i in circles[:]:
                    cv2.circle(grey, (i[0], i[1]), i[2], (255, 100, 100), 5)  # 画圆
                    cv2.circle(grey, (i[0], i[1]), 2, (255, 100, 100), 10)  # 画圆心
                for j in circles[:]:
                    yuanxinx = j[0]
                cv2.line(grey, (minyuanxinyuzhi, 0), (minyuanxinyuzhi, 480), (255, 100, 100), 2, 4)
                cv2.line(grey, (maxyuanxinyuzhi, 0), (maxyuanxinyuzhi, 480), (255, 100, 100), 2, 4)
                cv2.imshow('res',grey)               
                if yuanxinx >= minyuanxinyuzhi and yuanxinx <= maxyuanxinyuzhi:
                    print("forward")
                    #robot.movement.right_ward()
                elif yuanxinx >= maxyuanxinyuzhi:
                    print("turn right")
                    #robot.movement.left_ward()
                else:
                    print("turn left")
                    continue

            else:
                print("no circle")
                continue
        else:
            print("hough error")
    time.sleep(1)
    cap.release()
    cv2.destroyALLWindows()
    
    

if __name__ == '__main__':
    adjust()