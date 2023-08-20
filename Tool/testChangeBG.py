import os
import cv2
import numpy as np
# 读取txt
txtDir=r'D:\Desktop\Data\BUFF\Enhance\BLUE2000+RED400+BRIGHT'
txtNames = os.listdir(txtDir)
backDir= os.path.join(txtDir,"..",'Background')
backNames= os.listdir(backDir)
packPaths= [os.path.join(backDir,name) for name in backNames]

for name in txtNames:
    if ('-STATIC-' not in name) or ('.txt' not in name) or ('-STATIC-BRIGHT-' in name):
        continue
    
    txtPath = os.path.join(txtDir,name)

    with open(txtPath,'r') as f:
        for line in f.readlines():
            line=np.array(line.strip().split()[1:],dtype=np.float32).reshape(7,2)*np.array([640,384])
            while True:
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
            line=line.astype(np.int64)
            # line=np.array(line.strip().split()[1:],dtype=np.float32)
            print(txtPath)
            print(line)
            img = cv2.imread(txtPath.replace('.txt','.jpg'))
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
            
        cv2.rectangle(img,backline[0],backline[0]+backline[1],color=(255,255,255))
        pts=backline[2:]
        for i in range(len(pts)):
            background=cv2.line(background,pts[i],pts[(i+1)%5],color=(0,0,255),thickness=1)
        cv2.imshow("img",background)
        key=cv2.waitKey()
        if key == 27:
            break
cv2.destroyAllWindows()