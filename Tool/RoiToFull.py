import os
import cv2
import numpy as np

class RoiToFull:
    def __init__(self,txtDir,y1,y2,x1,x2):
        self.txtDir=txtDir
        self.broadCast=[y1,y2,x1,x2]
        self.shape=[y2-y1,x2-x1]
        
    # def transpose(self):
    #     pass
    def transMany(self):
        names = os.listdir(self.txtDir)
        for name in names:
            if os.path.splitext(name)[1]!='.txt':
                continue
            self.txtPath=os.path.join(self.txtDir,name)
            self.transOne()
        
    def transOne(self):
        # txtDir=self.txtDir
        imgSize=[1200,1920]
        y1,y2,x1,x2 = self.broadCast
        shape=[y2-y1,x2-x1]

        # label
        with open(self.txtPath,'r') as f:
            labels=np.asarray([line.strip().split()[1:] for line in f.readlines()],dtype=np.float64)
        
        # 取数
        if 'Full_' in self.txtPath:
            return
        
        newPath=self.txtPath[:-9]+'Full_'+self.txtPath[-9:]
        with open(newPath,'w') as f:
            for label in labels:
                x,y,w,h=label[:4]
                xywh=[
                    (x1+x*shape[1])/imgSize[1],
                    (y1+y*shape[0])/imgSize[0],
                    shape[1]*w/imgSize[1],
                    shape[0]*h/imgSize[0],
                    ]
                pts=label[4:].reshape(5,2)
                pts[:,0]=(x1+pts[:,0]*shape[1])/imgSize[1]
                pts[:,1]=(y1+pts[:,1]*shape[0])/imgSize[0]
                pts=sum(pts.tolist(),[])
                line = [str(i) for i in [0]+xywh+pts]
                line = " ".join(line)
                print((line))
                f.write(str(line))
                f.write('\n')
            
if __name__ == "__main__":
    path=r'D:\Desktop\Data\BUFF\版本控制\新一批400\400-Reflect'
    r=RoiToFull(path,300,900,0,960)
    r.transMany()