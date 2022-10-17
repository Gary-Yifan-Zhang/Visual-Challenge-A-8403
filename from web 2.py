"""
作者：杨一
此版代码为使用socket进行网络图像传输，并在算力较高的计算机平台上进行霍夫圆判断的代码。
目前处于只能勉强跑的屎山状态，但是其中的大部分函数和代码都可以直接应用到比赛中。
此代码（from web 2.py）为先运行的代码（接收端（高算力电脑）），运行完此代码后运行from web 1.py
from web 1.py是运行在机器人上的代码。
注意：ip地址接收端就是空的！！！
买一个好一点的路由器！！！！
"""



import socket
import time
import cv2
import numpy as np
width = 480
height = 180
MinRadius = 1
MaxRadius = 300
minyuanxinyuzhi = 180
maxyuanxinyuzhi = 300
def gaussian_mle(data):
    mu = data.mean(axis=0)
    var = (data-mu).T @ (data-mu) / data.shape[0] #  this is slightly suboptimal, but instructive
    return mu, var

def tf (x1,y1,r1):
    seita = 0
    l = 0
    tf_out = np.zeros(2)
    seita = x1/8
    seita = seita - 30
    print("seita = %f" % seita)
    l = 800000*((1/50)/15)/r1
    print("distence = %f" % l)
    tf_out[0] = seita
    tf_out[1] = l
    return tf_out
def ReceiveVideo():
    # IP地址'0.0.0.0'为等待客户端连接
    x_q = np.zeros(25)
    y_q = np.zeros(25)
    r_q = np.zeros(25)
    x_q_new = np.zeros(25)
    y_q_new = np.zeros(25)
    r_q_new = np.zeros(25)
    address = ('', 8002)
    # 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
    # socket.AF_INET：服务器之间网络通信
    # socket.SOCK_STREAM：流式socket , for TCP
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 将套接字绑定到地址, 在AF_INET下,以元组（host,port）的形式表示地址.
    s.bind(address)
    # 开始监听TCP传入连接。参数指定在拒绝连接之前，操作系统可以挂起的最大连接数量。该值至少为1，大部分应用程序设为5就可以了。
    s.listen(1)

    def recvall(sock, count):
        buf = b''  # buf是一个byte类型
        while count:
            # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    # 接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，可以用来接收和发送数据。addr是连接客户端的地址。
    # 没有连接则等待有连接
    conn, addr = s.accept()
    print('connect from:' + str(addr))

    while 1:
        start = time.time()  # 用于计算帧率信息
        length = recvall(conn, 16)  # 获得图片文件的长度,16代表获取长度
        stringData = recvall(conn, int(length))  # 根据获得的文件长度，获取图片文件
        data = np.frombuffer(stringData, np.uint8)  # 将获取到的字符流数据转换成1维数组
        decimg = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将数组解码成图像
        # cv2.imwrite("./test.jpg", decimg)
        # print(decimg)
        cv2.waitKey(1)
        # cv2.imshow('SERVER',decimg)#显示图像
        resized_height1 = int(width * 0.75)
        frame1 = cv2.resize(decimg, (width, resized_height1))
        hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)  # HSV空间
        lower_red = np.array([0, 40, 45])  # 设定红色的阈值
        upper_red = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)  # 设定取值范围
        res1 = cv2.bitwise_and(frame1, frame1, mask=mask)  # 对原图像处理
        cv2.imshow('res', hsv)
        # circles1 = cv2.HoughCircles(res1, cv2.HOUGH_GRADIENT, 1, 1000, param1=100, param2=30, minRadius=MinRadius,
        #                            maxRadius=MaxRadius)
        grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        circles1 = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1, 1000, param1=100, param2=30, minRadius=MinRadius,maxRadius=MaxRadius)

        if np.all(circles1) == 1:
            Radius = circles1[0, 0, 2]
            r_q_new = np.roll(r_q, 1)
            r_q_new[0] = Radius
            r_q = r_q_new
            r_final = gaussian_mle(r_q)
            print("radius = %f" % r_final[0])
            x = circles1[0, 0, 0]
            x_q_new = np.roll(x_q, 1)
            x_q_new[0] = x
            x_q = x_q_new
            x_final = gaussian_mle(x_q)
            print("x = %f" % x_final[0])
            y = circles1[0, 0, 1]
            y_q_new = np.roll(y_q, 1)
            y_q_new[0] = y
            y_q = y_q_new
            y_final = gaussian_mle(y_q)
            print("y = %f" % y_final[0])
            circles = circles1[0, :, :]  # 提取为2维
            circles = np.uint16(np.around(circles))  # 四舍五入，取整
            for i in circles[:]:
                cv2.circle(grey, (i[0], i[1]), i[2], (255, 100, 100), 5)  # 画圆
                cv2.circle(grey, (i[0], i[1]), 2, (255, 100, 100), 10)  # 画圆心
            for j in circles[:]:
                yuanxinx = j[0]
            cv2.line(grey, (minyuanxinyuzhi, 0), (minyuanxinyuzhi, 480), (255, 100, 100), 2, 4)
            cv2.line(grey, (maxyuanxinyuzhi, 0), (maxyuanxinyuzhi, 480), (255, 100, 100), 2, 4)
            cv2.imshow('res', grey)
            if yuanxinx <= minyuanxinyuzhi:
                print("右")
            elif yuanxinx >= maxyuanxinyuzhi:
                print("左")
            else:
                tf(x_final[0], y_final[0], r_final[0])
                # continue

        else:
            print("no circle")
            # continue


        # circles = circles1[0, :, :]  # 提取为2维
        # circles = np.unit16(np.around(circles))  # 四舍五入，取整
        # for i in circles[:]:
        #     cv2.circle(res1, (i[0], i[1]), i[2], (255, 100, 100), 5)  # 画圆
        #     cv2.circle(res1, (i[0], i[1]), 2, (255, 100, 100), 10)  # 画圆心
        # for j in circles[:]:
        #     yuanxinx = j[0]
        # cv2.line(res1, (minyuanxinyuzhi, 0), (minyuanxinyuzhi, 180), (0, 255, 0), 2, 4)
        # cv2.line(res1, (maxyuanxinyuzhi, 0), (maxyuanxinyuzhi, 180), (0, 255, 0), 2, 4)
        # if yuanxinx <= minyuanxinyuzhi:
        #     print("→")
        #     # robot.movement.right_ward()
        # elif yuanxinx >= maxyuanxinyuzhi:
        #     # robot.movement.left_ward()
        #     print("←")
        # else:
        #     break
        end = time.time()
        seconds = end - start
        fps = 1 / seconds;
        conn.send(bytes(str(int(fps)), encoding='utf-8'))
        # k = cv2.waitKey(10)&0xff
        # if k == 27:
        #    break
    s.close()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    ReceiveVideo()