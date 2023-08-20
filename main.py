# import json
import os,cv2,numpy as np
from detect_face import *
# from Camera.camera import Camera


class VideoDetector:
    def __init__(self,modelPath,videoPath):
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model = load_model(modelPath,self.__device)
        self.__cap = cv2.VideoCapture(videoPath)
        
    def run(self):
        cv2.namedWindow("rst",cv2.WINDOW_NORMAL)
        # getvideo = cv2.VideoWriter(r"E:\liantiaoall\smallban\AVISave\exp2best.avi",
        #                            cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 5, (640, 384))
        getvideo = cv2.VideoWriter(r"E:\fenqu\infantry\buff\usbuff3.avi",
                                   cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10, (640, 384))

        count = 0
        allimg = []
        while count < 4500:
            ret, img = self.__cap.read()
            if ret:
                # if 0 < count < 1100 or 1265 < count < 2560 or 4232 < count < 5437:
                img = cv2.resize(img, (640, 384))
                # getvideo.write(img)
                print()
                result=detect_one(self.__model, img, self.__device)
                getvideo.write(img)
                # if 200 < count < 500:
                # cv2.waitKey(1)
                allimg.append(result)
                print(count)
                count += 1
                if len(allimg) == 50:
                    allimg.remove(allimg[0])
                key = cv2.waitKey(0)
                if key == 115:      # s
                    if len(allimg) != 50:
                        i = 51 - len(allimg)
                    else:
                        i = 1
                    while True:
                        cv2.namedWindow("SB gtm", cv2.WINDOW_NORMAL)
                        cv2.imshow("SB gtm", allimg[50-i])
                        lastk = cv2.waitKey(1)
                        print("i", i)
                        if lastk == 97 and (50-i) > 0:     # a
                            i += 1
                        if lastk == 100 and 50-i < 48 and 50-i > 0:     # d
                            i -= 1
                        if lastk == 98:     # b
                            break
                if key == 27:
                    break

                cv2.imshow("rst", result)
            # key=cv2.waitKey(0)
            # if key == 27:
            #     break
        cv2.destroyAllWindows()


class CameraDetector:
    def __init__(self,modelPath,camera):
        self.__device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model=load_model(modelPath,self.__device)
        self.__camera=camera
        
    def run(self):
        while not isinstance(self.__camera.getFrame(), np.ndarray):
            pass
        cv2.namedWindow("rst",cv2.WINDOW_FREERATIO)
        while True:
            img=self.__camera.getFrame()
            img=cv2.resize(img,(640,384))
            img=detect_one(self.__model, img, self.__device)
            cv2.imshow("rst",img)
            key=cv2.waitKey(0)
            if key==27:
                break
        cv2.destroyAllWindows()
        self.__camera.stop()


class imgDetector:
    def __init__(self, modelPath):
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model = load_model(modelPath, self.__device)
        self.__path = "E:/National/buff/hashen/huangdong/"

    def run(self):
        cv2.namedWindow("rst", cv2.WINDOW_NORMAL)
        # getvideo = cv2.VideoWriter(r"E:\liantiaoall\smallban\AVISave\exp2best.avi",
        #                            cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 5, (640, 384))
        getvideo = cv2.VideoWriter(r"E:\fenqu\infantry\buff\usbuff3.avi",
                                   cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10, (640, 384))

        count = 0
        allimg = []
        for filename in os.listdir(self.__path):
            file_path = os.path.join(self.__path, filename)
            path = file_path
            print(self.__path, type(self.__path), filename)
            # if 0 < count < 1100 or 1265 < count < 2560 or 4232 < count < 5437:
            dst = cv2.imread(path)
            img = cv2.resize(dst, (640, 384))
            # getvideo.write(img)
            result = detect_one(self.__model, img, self.__device, self.__path, filename)
            # getvideo.write(img)
            cv2.imshow("rst", result)
            # if 200 < count < 500:
            # cv2.waitKey(1)
            allimg.append(result)
            print(count)
            count += 1
            if len(allimg) == 50:
                allimg.remove(allimg[0])
            key = cv2.waitKey(1)
            if key == 115:  # s
                if len(allimg) != 50:
                    i = 51 - len(allimg)
                else:
                    i = 1
                while True:
                    cv2.namedWindow("SB gtm", cv2.WINDOW_NORMAL)
                    cv2.imshow("SB gtm", allimg[50 - i])
                    lastk = cv2.waitKey(0)
                    print("i", i)
                    if lastk == 97 and (50 - i) > 0:  # a
                        i += 1
                    if lastk == 100 and (50 - i) < 48 and (50 - i) > 0:  # d
                        i -= 1
                    if lastk == 98:  # b
                        break
            if key == 27:
                break
            # key=cv2.waitKey(0)
            # if key == 27:
            #     break
        cv2.destroyAllWindows()


def showData(path,imgSize):
    """
    展示path目录下的jpg和txt标签图
    """
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    txtNames = [i for i in os.listdir(path) if ".txt" in i]
    for name in txtNames:
        imgPath=os.path.join(path,name.replace(".txt",".jpg"))
        txtPath=os.path.join(path,name)
        with open(txtPath,"r") as f:
            # img = cv2.imread(imgPath)
            img = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8),-1)
            
            # content = json.load(f)
            content=[[float(j) for j in i[1:]] for i in [line.strip().split() for line in f.readlines()]]
            # draw framework
            for data in content:
                xywh=(np.array(data[:4]).reshape(2,2)*imgSize)
                # xywh[1]+=np.array([10+np.random.random(),10+np.random.random()])
                xywh[0]-=(xywh[1]*0.5)
                xywh[1]+=xywh[0]
                xywh=xywh.astype(np.int64)  
                # print(xywh)
                pts=(np.array(data[4:]).reshape(5,2)*imgSize).astype(np.int64)
                # draw rect 
                img=cv2.rectangle(img,xywh[0],xywh[1],color=(0,0,255),thickness=1)
                for i in range(len(pts)):
                    img=cv2.line(img,pts[i],pts[(i+1)%5],color=(0,0,255),thickness=1)
                    
            cv2.imshow("img",img)
            print(imgPath)
            print(pts)
            key=cv2.waitKey(25)
            if key == 27:
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()
                           
# showData(r'D:\Desktop\Data\BUFF\Enhance\Parts\Little-Pure-Rename\BLUE',[640,384])


if __name__ == "__main__":
    # camera = Camera()
    # camera.start()
    # detector = VideoDetector(r"E:\RobotCenter\dataSet\buff\models\exp3\buffexp3.pt", r"E:\National\buff\3.avi")
    detector=imgDetector(r"E:\RobotCenter\dataSet\buff\models\exp4\buffexp4.pt")
    # detector = VideoDetector(r"E:\RobotCenter\dataSet\buff\models\best.pt", r"C:\Users\HZY\MVviewer\videos\A5201CU150_AM00990AAK00013\4.avi")
    # detector=VideoDetector(r"D:\Desktop\Code\Python\Projects\PTS5\Model\loss0.004\best.pt",r"D:\Desktop\Data\BUFF\Videos\1.avi")
    # detector=VideoDetector(r"Model\exp35\weights\last.pt",r"D:\Desktop\Data\BUFF\Videos\B-BLUE-STATIC-04.avi")
    # detector=CameraDetector(r"Model\exp35\weights\last.pt",camera)
    # detector=VideoDetector(r"Model\K80-2.pt",r"D:\Desktop\Data\BUFF\Videos\A-RED-STATIC-02.avi")
    
    detector.run()