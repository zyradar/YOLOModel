# 数据集分配
import os,random,shutil,json,numpy as np,cv2
from pandas import array
class DataSplitor:
    """
    path -> jpg+txt
    输出 -> --Src
            --DataX
                |--train        70
                |   |--images   
                |   |--labels   
                |--val          20
                |--test         10
    """
    def __init__(self,path:str,ratio:list or tuple = (0.8,0.1,0.1)):
        self._path=path
        self._ratio=ratio
        self._setNames=("train","val","test")
        self._outDirName="Datax"

    def run(self):
        # self.__transLabels()
        self._split()
        
    def _split(self):
        fileNames=os.listdir(self._path)
        labNames=[x for x in fileNames if ".txt" in x]
        # print(imgNames)
        random.shuffle(labNames)
        shuffleNames=labNames
        
        output=[]   #[list*3]
        length=len(shuffleNames)
        for ratio in self._ratio:
            num=int(ratio*length)
            names=shuffleNames[:num]
            shuffleNames=shuffleNames[num:]
            output.append(names)
        
        if len(shuffleNames):
            output[0].extend(shuffleNames)
            
        # 文件操作
        # 创建DataX
        outDirPath=os.path.join(self._path,"..",self._outDirName)
        
        if os.path.exists(outDirPath):
            shutil.rmtree(outDirPath)
        os.mkdir(outDirPath)
            
        for i,names in enumerate(output):
            dirName=self._setNames[i]
            os.mkdir(os.path.join(outDirPath,dirName))
            dirImg=os.path.join(outDirPath,dirName,"images")
            dirLab=os.path.join(outDirPath,dirName,"labels")
            os.mkdir(dirImg)
            os.mkdir(dirLab)
            
            # images/labels
            for name in names:
                srcLab = os.path.join(self._path,name)
                dstLab = os.path.join(dirLab,name)
                srcImg = os.path.join(self._path,name.replace(".txt",".jpg"))
                dstImg = os.path.join(dirImg,name.replace(".txt",".jpg"))
                shutil.copyfile(srcImg,dstImg)
                shutil.copyfile(srcLab,dstLab)
                
class JsonToTxt:
    
    def __init__(self,txtDir):
        self._path=txtDir
        self._cNames=["0"]
        self._imgSize=[640,384]
        self._bias=3
        
    def run(self):
        self._transLabels()
    
    def _transLabels(self):
        # tempDir=os.path.join(self._path, "..",".labels")
        # if os.path.exists(tempDir):
        #     shutil.rmtree(tempDir)
        # os.mkdir(tempDir)
        for name in [x for x in os.listdir(self._path) if ".json" in x]:
            with open(os.path.join(self._path,name),'r') as file:
                output=self._read_json_to_face(file)
                
            with open(os.path.join(self._path,name).replace('.json','.txt'),'w') as file:
                for line in output:
                    file.write(line+'\n')
                        
    def _read_json_to_face(self,file):
        output=[]
        info= json.load(file)
        shapes=info.get("shapes")
        for x in shapes:
            label=str(self._cNames.index(x.get("label")))
            pts=x.get("points")
            # rect = generalRect.boxForPoint(pts,self._imgSize,self._bias)#->str
            rect = generalRect.boxForPoints(pts,5,self._imgSize)#->str
            pts=" ".join([str(i) for i in (sum((np.array(pts,dtype=np.float64)/self._imgSize).tolist(),[]))])
            # 标签 x y w h p1x p2x---
            string =" ".join([label,rect,pts])
            output.append(string)
        return output
    
class generalRect:
    
    def boxForPoints(pts:list[list],pointNum:int,imgSize:list[int],extend=12):
        """
        imgSize:[x,y]
        """
        fx=1000000
        # str转numpy.int64(cv外接矩形)
        src=np.array(pts[:pointNum])*fx     #转numpy放大
        src=np.array(src,dtype=np.int64)    #转int64
        rect=cv2.boundingRect(src)          #xywh
        rect=np.array(rect,dtype=np.float64)/fx
        
        rect=rect.reshape((2,2))
        # rect[1]/=2
        # rect[0]+=rect[1]
        rect[0]+=rect[1]*0.5
        rect[1]+=[extend+np.random.random(),extend+np.random.random()]
        rect=(rect/imgSize).tolist()    #归一化
        rect=[str(x) for x in sum(rect,[])]
        string = " ".join(rect)
        return string
    
    def boxForPoint(pts:list,imgSize:list[int],bias:float):
        if not len(pts)==5:
            print("[INFO]::非五点不适用")

        src=np.array(pts)
        xy=[str(x) for x in ((src[-1]-bias).clip(min=0)/imgSize).tolist()]
        wh=[str(x) for x in (array([bias*2]*2)/imgSize).tolist()]
        string = " ".join(xy+wh)
        
        return string
    
txtDir=r"D:\Desktop\Data\BUFF\Enhance\Parts\Little-Pure-Rename\GATHER"
# txtDir=r"D:\Desktop\Data\BUFF\Temp\RED-900"

# jsonToTxt=JsonToTxt(txtDir)
# jsonToTxt.run()

dataSplitor=DataSplitor(txtDir,(0.8,0.2,0))
dataSplitor.run()