import cv2
import numpy as np
#import pygame
import cv2 as cv
import time
#import smtplib
from matplotlib import pyplot as plt

#im = cv2.imread('E:/des/Civil/Patholenew/1 (27).jpg')
#im = cv2.imread('E:/des/Civil/Nopatholenew/2 (55).jpg')
# CODE TO CONVERT TO GRAYSCALE
im=cv2.imread("E:/des/Civil/2.jpg")

gray1 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# save the image
cv2.imwrite('graypothholeresult.jpg', gray1)
#CONTOUR DETECTION CODE
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)

contours1, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
contours2, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#img1 = im.copy()
img2 = im.copy()

#out = cv2.drawContours(img1, contours1, -1, (255,0,0), 2)
out = cv2.drawContours(img2, contours2, -1, (250,250,250),1)
#out = np.hstack([img1, img2])
cv2.imshow('img1',img2)
cv2.waitKey(0)
plt.subplot(331),plt.imshow(im),plt.title('GRAY')
plt.xticks([]), plt.yticks([])
#img = cv2.imread('E:/des/Civil/Patholenew/1 (27).jpg',0)
#img = cv2.imread('E:/des/Civil/Nopatholenew/2 (55).jpg',0)
img=cv2.imread("E:/des/Civil/2.jpg")
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2) 
cnt = contours[0]
M = cv2.moments(cnt)



#print M
perimeter = cv2.arcLength(cnt,True)
#print perimeter
area = cv2.contourArea(cnt)
#print area
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
#print epsilon
#print approx
for c in contours:
    rect = cv2.boundingRect(c)
    if rect[2] < 100 or rect[3] < 100: continue
    #print cv2.contourArea(c)
    x,y,w,h = rect
    cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),8)
    cv2.putText(img2,'Moth Detected',(x+w+40,y+h),0,2.0,(0,255,0))
cv2.imshow("Show",img)
cv2.waitKey()  
cv2.destroyAllWindows()
k = cv2.isContourConvex(cnt)

#to check convexity
print(k)



#blur
blur = cv2.blur(im,(5,5))
#guassian blur 
gblur = cv2.GaussianBlur(im,(5,5),0)
#median 
median = cv2.medianBlur(im,5)
#erosion
kernel = np.ones((4,4),np.uint8)
erosion = cv2.erode(median,kernel,iterations = 1)
dilation = cv2.dilate(erosion,kernel,iterations = 5)
#erosion followed dilation
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
#canny edge detection
edges = cv2.Canny(dilation,9,220)

kernel2 = np.ones((50,50),np.uint8)
closing2 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel2)



#plotting using matplotlib
plt.subplot(331),plt.imshow(blur),plt.title('BLURRED')
plt.xticks([]), plt.yticks([])
plt.subplot(332),plt.imshow(gblur),plt.title('guassianblur')
plt.xticks([]), plt.yticks([])        
plt.subplot(333),plt.imshow(median),plt.title('Medianblur')
plt.xticks([]), plt.yticks([]) 
plt.subplot(335),plt.imshow(img,cmap = 'gray')
plt.title('Dilated Image'), plt.xticks([]), plt.yticks([])
plt.subplot(337),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(334),plt.imshow(erosion),plt.title('EROSION')
plt.xticks([]), plt.yticks([])
plt.subplot(336),plt.imshow(closing),plt.title('Morphology(closing)')
plt.xticks([]), plt.yticks([])
plt.subplot(338),plt.imshow(closing2,cmap = 'gray'),plt.title('Detected Area')
plt.xticks([]), plt.yticks([])
plt.show()



# In[2]:

import cv2
import numpy as np
#import pygame
import time
import smtplib
from matplotlib import pyplot as plt
import glob
    
images_path1="E:/des/Civil/Nopatholenew/"

images1=glob.glob(images_path1+"*.jpg")

# read images and append to list
nopath_images_original=[]


for i in images1:
    readImage=cv2.imread(i,0)
    readImage=cv2.resize(readImage,(250,250))
    nopath_images_original.append(readImage)

print('Reading of No pathole Images Done')




images_path4="E:/des/Civil/Patholenew/"

images4=glob.glob(images_path4+"*.jpg")

pathole_images_original=[]


for a in images4:
    readImage=cv2.imread(a,0)
    readImage=cv2.resize(readImage,(250,250))
    pathole_images_original.append(readImage)

print("Reading of pathole Images Done")



kernel = np.ones((3,3),np.uint8)
outn_array=[]
nedges_array=[]
nfinal1_array=[]
nfinal_array=[] 

for c in nopath_images_original:
    retn,threshn = cv2.threshold(c,127,255,0)
    contoursn, hierarchyn = cv2.findContours(threshn,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    outn= cv2.drawContours(c, contoursn, -1, (250,250,250),1)
    nmedian = cv2.medianBlur(c,5)
    nerosion = cv2.erode(outn,kernel,iterations = 1)
    ndilation = cv2.dilate(nerosion,kernel,iterations = 5)
    nclosing = cv2.morphologyEx(ndilation, cv2.MORPH_CLOSE, kernel)
    #nedges = cv2.Canny(ndilation,150,220)
    nedges = cv2.Canny(nclosing,150,220)
    kernel2 = np.ones((50,50),np.uint8)
    nclosing2 = cv2.morphologyEx(nedges, cv2.MORPH_CLOSE, kernel2)
    
    outn_array.append(outn)
    nedges_array.append(nedges)
    nfinal1_array.append(nclosing)
    nfinal_array.append(nclosing2)

    


kernel = np.ones((3,3),np.uint8)
out_array=[] 
edges_array=[]
final1_array=[]
final_array=[] 

for d in pathole_images_original:
    ret,thresh = cv2.threshold(d,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    out= cv2.drawContours(d, contours, -1, (250,250,250),1)
    median = cv2.medianBlur(d,5)
    erosion = cv2.erode(median,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 5)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    #edges = cv2.Canny(dilation,9,220)
    edges = cv2.Canny(closing,150,220)
    kernel2 = np.ones((50,50),np.uint8)
    closing2 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel2)
  
    out_array.append(out)
    edges_array.append(edges)
    final1_array.append(closing)
    final_array.append(closing2)


cv2.imshow("Show",nfinal_array[9])
cv2.waitKey()  
cv2.destroyAllWindows()


# In[3]:
    
import statistics

ncountmean_array=[]
for n in nfinal_array:
    count=0
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            if n[i,j]>0:
                count=count+1
    ncountmean_array.append(count) 


t=0
f=-1
for n in ncountmean_array:
    f=f+1
    t=t+ncountmean_array[f]
print(t/72)
nmedian=statistics.median(ncountmean_array)
print(statistics.median(ncountmean_array))    




countmean_array=[]
for p in final_array:
    count2=0
    for i in range(p.shape[0]):
       for j in range(p.shape[1]):
            if p[i,j]>0:
                count2=count2+1
    countmean_array.append(count2) 

t=0
f=-1
for n in countmean_array:
    f=f+1
    t=t+countmean_array[f]
print(t/60)
median=statistics.median(countmean_array)
print(statistics.median(countmean_array))



f=-1
wrong=0 
for n in ncountmean_array:
    f=f+1
    if ncountmean_array[f]>((nmedian+median)/2):
        wrong=wrong+1
print("accuracy of no pathole detection=", (100-(wrong/72)*100))


f=-1
wrong=0 
for n in countmean_array:
    f=f+1
    if countmean_array[f]<((nmedian+median)/2):
        wrong=wrong+1
print("accuracy of pathole detection=", (100-(wrong/60)*100))
    