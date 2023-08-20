# 检验导入路径
import sys, os

from Mouse import Keypoly

__presentDir = os.path.join(os.getcwd(), os.path.dirname(__file__))
sys.path.insert(0, __presentDir)
from tracking.detection.yolo_2.yolo import mathdetect
from cameradetect import *
from Serial.myserial import *
from track import *


class Camera(Thread):
    def __init__(self, serial=None, track=None):
        Thread.__init__(self, name="Camera")
        self.track = track
        self.serial = serial
        self.frame = np.zeros(())
        self.isLoop = False
        self.streamsource = None
        self.printFPS = True
        self.exposureTime = 16000.0
        self.gainRaw = 1.4
        self.gamma = 0.6
        self.frame = np.zeros(())
        self.flags = dict.fromkeys(['fly', 'rock', 'buff', 'areaA', 'areaB'], False)
        self.dst = np.zeros(())
        self.img = np.zeros(())
        self.t1 = 0
        self.enemycolor = None
        self.yolo = mathdetect()

        # self.__openCamera()

    def run(self):
        print("[INFO]::Thread-Camera running")
        self.process()

    """def verifyCamera(self):
        if isinstance(self.Camera,Camera):
            if isinstance(self.__camera.getFrame(),numpy.ndarray):
                return True
            else:
                return False
        else:
            return False"""

    def setFlags(self):
        self.contesttime=self.serial.gettime()
        if 405 < self.contesttime < 408:
           self.flags["rock"] = True
        if 360 < self.contesttime < 375 or 180 < self.contesttime < 195:
           self.flags["buff"] = True

    def setEnemyColor(self):
        default = "R"
        if isinstance(self.serial, Serial):
            if isinstance(self.serial.getenemyColor(), str):
                self.enemycolor = self.serial.getenemycolor()
                print("[INFO]::Locator成功获取敌方颜色信息 {:s}".format(self.enemycolor))
            else:
                self.enemycolor = default
                print("[INFO]::Locator使用默认敌方颜色信息 {:s}".format(self.enemycolor))
        else:
            self.enemycolor = default
            print("[INFO]::Locator使用默认敌方颜色信息 {:s}".format(self.enemycolor))

    def process(self):
        self.setEnemyColor()
        self.isLoop = True
        self.setFrame()

    def stop(self):
        self.isLoop = False
        if isinstance(self.serial, Serial):
            self.serial.stop()
        print("[INFO]::Thread-Locator stopped")

    def setFrame(self):
        CNT = 0
        enemy = {'B': (255, 0, 0),
                 'R': (0, 0, 255)}
        mouse_position = []
        isgetvideo = False
        VID = True
        path = "C:/Users/HZY/Pictures/Saved.avi"
        map = cv2.imread("E:/myradar/Location/map.png")
        map = cv2.resize(map, (679, 1229))
        h2, w2, c2 = map.shape[:]
        map = cv2.resize(map, (w2 // 2, h2 // 2))
        cap = cv2.VideoCapture(path)
        fps = 30
        size = (1920, 1200)
        getvideo = cv2.VideoWriter("C:/Users/HZY/Pictures/handelSaved.avi",
                                   cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
        xxyys = poly_maxmin_xy(all_region)
        keymouse = Keypoly(mouse_position)
        det = Yolo2Tracker(need_speed=True, need_angle=True)
        cnnt = 0
        cout = 1
        while self.isLoop:
            if isinstance(self.serial, Serial):
                self.setFlags()
            else:
                pass
            self.t1 = time.time()
            if VID:
                ret, self.img = cap.read()
            elif VID is False:
                if CNT == 0:
                    map_test = map.copy()
                    cvImage, nRet, streamSource, camera, CNT, VID = demo(None, None, None, None, CNT)
                    if VID:
                        print("is open video!")
                        continue
                else:
                    cvImage, nRet, streamSource, camera, CNT = demo(cvImage, nRet, streamSource, camera, CNT)
                self.dst = np.zeros((1200, 1920 * CNT, 3), dtype=np.uint8)
                self.img = np.zeros((1200, 1920 * CNT, 3), dtype=np.uint8)
                for i in range(0, len(cvImage)):
                    self.dst[0:1200, 1920 * i:1920 * (i + 1)] = cvImage[i]
                    if CNT == 2:
                        # img = None
                        src1 = self.dst[0:1200, 3520:3840]
                        src2 = self.dst[0:1200, 0:1920]
                        self.img[0:1200, 0:1920] = src2
                        self.img[0:1200, 1920:2240] = src1
                    if CNT == 1:
                        self.img = self.dst
                cvImage.clear()
            self.img = self.img[432:1200, 0:1920]
            cnnt += 1
            map_test = map.copy()
            h1, w1, c1 = self.img.shape[:]
            self.img = cv2.resize(self.img, (w1 // 2, h1 // 2))
            self.frame = self.img
            k = cv2.waitKey(50)
            if keymouse.isfinish:
                robot_point, img = self.yolo.detect(self.frame)
                # img, robot_point = det.deal_one_frame(self.frame, 30, True)
                print(robot_point)
                """通讯发送识别位置信息"""
                if robot_point:
                    new_enemy = map_new(map_test, robot_point, all_region, xxyys, enemy, keymouse.M)
                    if isinstance(self.serial, Serial):
                        for point in new_enemy:
                            """需调整"""
                            self.serial.sendLocation(cout % 5, point[0] / map_w, point[1] / map_h)
                            cout += 1
                    cv2.imshow("cap", self.frame)
                    cv2.imshow("map", map_test)
            else:
                keymouse.keyboard(k, self.frame)
                keymouse.mouse()
                keymouse.viwe()
            if k == 27:
                if VID is False:
                    nRet = shutdown(cvImage, nRet, streamSource, camera, CNT)
                    if nRet != 0:
                        print("Some Error happend")
                    print("--------- Demo end ---------")
                else:
                    cap.release()
                    cv2.destroyAllWindows()
                break
        # 0.5s exit
        time.sleep(0.5)


if __name__ == "__main__":
    x = Camera()
    x.start()

    x.join()

