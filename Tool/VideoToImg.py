# 图像处理
import os
import cv2
def VideoToImg(videoPath):
    """
    将一个视频跳帧转换成图片（640*384）
    """
    cap = cv2.VideoCapture(videoPath)
    videoName=os.path.splitext(videoPath)[0]
    # 检测count
    count=0
    while True:
        imgPath="{:s}_{:0>5d}{:s}".format(videoName,count,".jpg")
        if not os.path.isfile(imgPath):
            break
        count+=1
        
    while True:
        ret,frame=cap.read()
        
        if ret:
            imgPath="{:s}_{:0>5d}{:s}".format(videoName,count,".jpg")
            # frame=frame[300:300+600,:960]
            img = cv2.resize(frame,(640,384))
            # 跳帧
            # if True:
            if not count%3:
                cv2.imwrite(imgPath,img)
            count+=1
        else:
            # print(count)
            break
     
if __name__ == "__main__":
        
    videoDir=r"D:\Desktop\Data\BUFF\Temp"
    # 'A-BLUE-STATIC-01.avi'
    # 获取目录文件名
    names = os.listdir(videoDir)
    # for name in [name for name in names if (name.find(".avi"))]:
    #     os.path.splitext(name)
    for name in names:
        if not name.find(".avi"):
            continue
        VideoToImg(os.path.join(videoDir,name))