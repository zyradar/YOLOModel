import os
import cv2
import shutil
import numpy as np

class Enhancer:
    
    def applyAll(txtPath):
        Enhancer.changeBackground(txtPath)
        # Enhancer.generateBrightImage(txtPath)
        # Enhancer.changeBackground(txtPath)
        
    def contrast_brightness_image(src1, a, g):
        h, w, ch = src1.shape # 获取shape的数值，height、width和通道

        # 新建全零图片数组src2，将height、width，类型设置为原图片的通道类型（色素全为零，输出为全黑图片）
        src2 = np.zeros([h, w, ch], src1.dtype)
        dst = cv2.addWeighted(src1, a, src2, 1-a, g)
        # cv.namedWindow("con-bri-demo",cv.WINDOW_NORMAL)
        # cv2.imshow("con-bri-demo", dst)
        return dst

    def augment_hsv(img, hgain=0, sgain=0, vgain=-0.5):
    # BLUE to RED(ORANGE)
    # def augment_hsv(img, hgain=-0.4, sgain=-.5, vgain=0.5):
    # BLUE to RED
    # def augment_hsv(img, hgain=-0.2, sgain=.2, vgain=0.5):
        r = np.array([hgain, sgain, vgain]) + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
    
    def generateBrightImage(txtDir):
        names = os.listdir(txtDir)
        for name in names:
            if ('-STATIC-' not in name) or ('.txt' not in name) or ('-BRIGHT-' in name):
                continue

            txtPath=os.path.join(txtDir,name)
            imgPath=txtPath.replace('.txt','.jpg')
            brightTxtPath=txtPath.replace('-STATIC-','-STATIC-BRIGHT-')
            brightImagePath=imgPath.replace('-STATIC-','-STATIC-BRIGHT-')
            
            img = cv2.imread(imgPath)
            img = Enhancer.contrast_brightness_image(img,1.2,-100)
            # img = Enhancer.contrast_brightness_image(img,np.random.random()*0.2+0.9,np.random.random()*100-50)
            # img = Enhancer.contrast_brightness_image(img,np.random.random()*0.2+1,np.random.random()*30-100)
            cv2.imwrite(brightImagePath,img)
            shutil.copyfile(txtPath,brightTxtPath)
    
    def changeBackground(txtDir):
        # 读取txt
        # txtDir=r'D:\Desktop\Data\BUFF\Enhance\BLUE2000+RED400+BRIGHT'
        txtNames = os.listdir(txtDir)
        backDir= os.path.join(txtDir,"..",'Background')
        backNames= os.listdir(backDir)
        packPaths= [os.path.join(backDir,name) for name in backNames]

        for name in txtNames:
            if  ('.txt' not in name) or ('-BACK-' in name):
                continue
            
            txtPath = os.path.join(txtDir,name)
            newLines=[]
            with open(txtPath,'r') as f:
                for line in f.readlines():
                    line=np.array(line.strip().split()[1:],dtype=np.float64).reshape(7,2)*np.array([640,384])
                    while True:
                        # ----------------------------------
                        # 放缩
                        # ----------------------------------
                        # For BIG
                        # bias=(np.random.random()*0.3)+0.8
                        # For LITTLE
                        bias=(np.random.random()*0.2)+1
                        
                        if True not in (((line[0]-(line[1]*bias*0.5)))<[0]*2) and True not in (((line[0]+line[1]*bias*0.5)>[640,384])):
                            break
                    newSize=(line[1]*bias).astype(np.int64)
                    newPoint1=line[0]-newSize*0.5
                    
                    # line[0]=newPoint1
                    # line[1]=newSize
                    line[0]-=line[1]*0.5
                    line[2:]=((line[2:]-line[0])/line[1])*newSize+newPoint1
                    # line[0]=newPoint1
                    # line[1]=newSize
                    # line=line.astype(np.int64)
                    # line=np.array(line.strip().split()[1:],dtype=np.float32)
                    # print(txtPath)
                    # print(line)
                    img = cv2.imread(txtPath.replace('.txt','.jpg'))
                    # Enhancer.augment_hsv(img)
                    
                    randomint=np.random.randint(len(packPaths))
                    background = cv2.imread(packPaths[randomint])

                    while True:
                        bias=np.array([np.random.randint(-320,320),np.random.randint(-192,192)])
                        # bias=np.array([0,0])
                        if (True not in (newPoint1+bias <[0]*2)) and (True not in (newPoint1+newSize+bias >[640,384])):
                            break
                    backline=line.copy()
                    backline[0]=newPoint1
                    backline[1]=newSize
                    
                    backline+=bias
                    backline[1]-=bias
                    
                    backlineInt=backline[:2].astype(np.int64)
                    lineInt=line[:2].astype(np.int64)
                    
                    background[backlineInt[0][1]:backlineInt[0][1]+backlineInt[1][1],backlineInt[0][0]:backlineInt[0][0]+backlineInt[1][0]]=\
                            cv2.resize(img[lineInt[0][1]:lineInt[0][1]+lineInt[1][1],lineInt[0][0]:lineInt[0][0]+lineInt[1][0]],newSize)
                            
                    backline[0]+=backline[1]/2
                    newLines.append(" ".join(str(i) for i in [0]+sum((backline/[640,384]).tolist(),[])))
                    
            newTxtPath=txtPath.replace('IC-','IC-BACK-')
            with open(newTxtPath,'w') as f:
                for newLine in newLines:
                    f.write(newLine+'\n')
            cv2.imwrite(newTxtPath.replace('.txt','.jpg'),background)        

txtPath=r'D:\Desktop\Data\BUFF\Enhance\Parts\Little-Pure-Rename\BLUE'
Enhancer.applyAll(txtPath)