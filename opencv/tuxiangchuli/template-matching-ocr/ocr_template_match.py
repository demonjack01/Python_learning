import argparse
from tokenize import group
from unittest import result
import cv2 
import numpy as np
import myutils
from  imutils import contours

ap=argparse.ArgumentParser()
ap.add_argument("--image",required=True,help="path to input image")
ap.add_argument("--template",required=True,help="path to template OCR-A image")
args=vars(ap.parse_args())
print(args)

FIRST_NUMBER={
    "3":"American Express",
    "4":"Visa",
    "5":"MasterCard",
    "6":"Discover Card"
}

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img=cv2.imread(args["template"])
cv_show('img',img)

ref=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv_show('ref',ref)

ref=cv2.threshold(ref,10,255,cv2.THRESH_BINARY_INV)[1]
cv_show('ref',ref)

refCnts,hierarchy=cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,refCnts,-1,(0,0,255),3)
cv_show('img',img)
print(np.array(refCnts).shape)
refCnts=contours.sort_contours(refCnts,method="left-to-right")[0]
digits={}

for (i,c) in enumerate(refCnts):
    (x,y,w,h)=cv2.boundingRect(c)
    roi=ref[y:y+h,x:x+w]
    roi=cv2.resize(roi,(57,88))

    digits[i]=roi

rectKernal=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernal=cv2.getStructuringElement(cv2.MORPH_RECT,(6,6))

#step 读取灰度图
image=cv2.imread(args["image"])
cv_show('image',image)
image=myutils.resize(image,width=300)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)

#step 礼帽操作
tophat=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernal)   #! 礼帽操作，突出更明亮的区域
cv_show('tophat',tophat)

#step 梯度处理、归一化
gradX=cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
gradX=np.absolute(gradX)
(minval,maxval)=(np.min(gradX),np.max(gradX))
gradX=(255*(gradX-minval))/(maxval-minval)
gradX=gradX.astype("uint8")

print(np.array(gradX).shape)
cv_show('gradX',gradX)

#step 通过闭操作将数字连在一起
gradX=cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernal)
cv_show('gradX',gradX)

#step 阈值操作
thresh=cv2.threshold(gradX,0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
cv_show('thresh',thresh)

#step 再来个闭操作
thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernal)
cv_show('thresh',thresh) 

#step 计算轮廓
threshCnts,hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cnts=threshCnts
cur_img=image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),2)
cv_show('img',cur_img)
locs=[]

#step 遍历轮廓
for (i,c) in enumerate(cnts):
    (x,y,w,h)=cv2.boundingRect(c)
    ar=w/float(h)

    if ar>2.5 and ar<4.0:
        if (w>40 and w<55) and (h>10 and h<20):
            locs.append((x,y,w,h))

locs=sorted(locs,key=lambda x:x[0])
output=[]

#step 遍历每一个轮廓中的数字
for (i,(gX,gY,gW,gH)) in enumerate(locs):
    groupOutput=[]

    group=gray[gY-5:gY+gH+5,gX-5:gX+gW+5]
    cv_show('group',group)
    #step 预处理
    group=cv2.threshold(group,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    cv_show('group',group)

    digitCnts,hierarchy=cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts=contours.sort_contours(digitCnts,method="left-to-right")[0]

    for c in digitCnts:
        (x,y,w,h)=cv2.boundingRect(c)
        roi=group[y:y+h,x:x+w]
        roi=cv2.resize(roi,(57,88))
        cv_show('roi',roi)

        #step 计算匹配得分 
        scores=[]
        for (digit,digitROI) in digits.items():
            result=cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF)
            (_,score,_,_)=cv2.minMaxLoc(result)
            scores.append(score) 
        
        groupOutput.append(str(np.argmax(scores)))

    cv2.rectangle(image, (gX - 5, gY - 5),(gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    
    output.extend(groupOutput)

print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
