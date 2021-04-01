# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 09:42:12 2019

@author: MaryamHashemi
"""

#importing Required Libraries to run code
import glob
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.feature import hog


# In[1]:

images_path1="E:/des/Civil/Nopatholenew/"
#images_path1="E:/des/Civil/no pothole/no pothole/"
#images_path2="E:/des/Civil/no pothole(2)/no pothole/Asphalt Pavement/"
#images_path3="C:/Users/MaryamHashemi/Desktop/Civil/no pothole(2)/no pothole/Road Marking/"

images1=glob.glob(images_path1+"*.jpg")
#images2=glob.glob(images_path2+"*.jpg")
#images3=glob.glob(images_path3+"*.jpg")


images=[images1]



#reading image paths with glob

# read images and append to list
nopathole_images_original=[]
for i in images:
    for imagePath in i:
        readImage=cv2.imread(imagePath)
        readImage=cv2.resize(readImage,(250,250))
#    rgbImage = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
        nopathole_images_original.append(readImage)




images_path4="E:/des/Civil/Patholenew/"
#images_path4="E:/des/Civil/Pothole/Pothole/"
#images_path5="E:/des/Civil/Pothole/Pothole/"

images4=glob.glob(images_path4+"*.jpg")
#images5=glob.glob(images_path5+"*.jpg")

images_normal=[images4]

pathole_images_original=[]


for a in images_normal:
    for imagePath in a:
        readImage=cv2.imread(imagePath)
        readImage=cv2.resize(readImage,(250,250))
#    rgbImage = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
        pathole_images_original.append(readImage)
        
print("Reading Images Done")


# In[2]:


# Visualizing the Pathole and Non pathole Images

f, axes = plt.subplots(4,2, figsize=(10,10))
plt.subplots_adjust(hspace=0.5)

for index in range(4):
    pathole=random.randint(0, len(pathole_images_original)-1)
    nopathole=random.randint(0, len(nopathole_images_original)-1)
    axes[index,0].imshow(pathole_images_original[pathole])
    axes[index,0].set_title("defect")
    axes[index,1].imshow(nopathole_images_original[nopathole])
    axes[index,1].set_title("Non defect")
print("Shape of defect Image" +  str(pathole_images_original[pathole].shape))
print("Shape of Non defect Image" +  str(nopathole_images_original[nopathole].shape))





### Extract Color Space

#creating a Histogram
def ExtractColorHistogram(image, nbins=32, bins_range=(0,255), resize=None):
    if(resize !=None):
        image= cv2.resize(image, resize)
    zero_channel= np.histogram(image[:,:,0], bins=nbins, range=bins_range)
    first_channel= np.histogram(image[:,:,1], bins=nbins, range=bins_range)
    second_channel= np.histogram(image[:,:,2], bins=nbins, range=bins_range)
    return zero_channel,first_channel, second_channel

#Find Center of the bin edges
def FindBinCenter(histogram_channel):
    bin_edges = histogram_channel[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    return bin_centers

#Extracting Color Features from bin lengths
def ExtractColorFeatures(zero_channel, first_channel, second_channel):
    return np.concatenate((zero_channel[0], first_channel[0], second_channel[0]))






# Checking Color Features for pathole

f, axes= plt.subplots(4,5, figsize=(20,10))
f.subplots_adjust(hspace=0.5)

for index in range(4):
    
    pathole=random.randint(0, len(pathole_images_original)-1)
    nopathole=random.randint(0, len(nopathole_images_original)-1)
    
    coloredImage= cv2.cvtColor(pathole_images_original[pathole],cv2.COLOR_RGB2YUV)
    r,g,b = ExtractColorHistogram(coloredImage,128)
   
    center= FindBinCenter(r)
    axes[index,0].imshow(pathole_images_original[pathole])
    axes[index,0].set_title("defect")
    axes[index,1].set_xlim(0,256)
    axes[index,1].bar(center,r[0])
    axes[index,1].set_title("Y")
    axes[index,2].set_xlim(0,256)
    axes[index,2].bar(center,g[0])
    axes[index,2].set_title("U")
    axes[index,3].set_xlim(0,256)
    axes[index,3].bar(center,b[0])
    axes[index,3].set_title("V")
    axes[index,4].imshow(coloredImage)
    axes[index,4].set_title("YUV colorspace")
    
features = ExtractColorFeatures(r,g,b)
print("No of features are "+ str(len(features)))











# Checking Color Features for Non pathole

f, axes= plt.subplots(4,5, figsize=(20,10))
f.subplots_adjust(hspace=0.5)

for index in range(4):
    nopathole=random.randint(0, len(nopathole_images_original)-1)
    coloredImage= cv2.cvtColor(nopathole_images_original[nopathole],cv2.COLOR_RGB2YUV)
    r,g,b = ExtractColorHistogram(coloredImage)
    
    center= FindBinCenter(r)
    axes[index,0].imshow(nopathole_images_original[nopathole])
    axes[index,0].set_title("Non defect")
    axes[index,1].set_xlim(0,256)
    axes[index,1].bar(center,r[0])
    axes[index,1].set_title("Y")
    axes[index,2].set_xlim(0,256)
    axes[index,2].bar(center,g[0])
    axes[index,2].set_title("U")
    axes[index,3].set_xlim(0,256)
    axes[index,3].bar(center,b[0])
    axes[index,3].set_title("V")
    axes[index,4].imshow(coloredImage)
    axes[index,4].set_title("YUV colorspace")
    
    
    

# In[3]:
    
#Resizing Image to extract features, so as to reduce the feature vector size
def SpatialBinningFeatures(image,size):
    image= cv2.resize(image,size)
    return image.ravel()    


#testing the spatial binning (reducing feature size, only important features remain)

featureList=SpatialBinningFeatures(pathole_images_original[1],(16,16))
print("No of features before spatial binning",len(pathole_images_original[1].ravel()))
print("No of features after spatial binning",len(featureList))



# General method to extact the HOG of the image

def GetFeaturesFromHog(image,orient,cellsPerBlock,pixelsPerCell, visualise= False, feature_vector_flag=True):
    if(visualise==True):
        hog_features, hog_image = hog(image, orientations=orient,
                          pixels_per_cell=(pixelsPerCell, pixelsPerCell), 
                          cells_per_block=(cellsPerBlock, cellsPerBlock), 
                          visualise=True, feature_vector=feature_vector_flag)
        return hog_features, hog_image
    else:
        hog_features = hog(image, orientations=orient,
                          pixels_per_cell=(pixelsPerCell, pixelsPerCell), 
                          cells_per_block=(cellsPerBlock, cellsPerBlock), 
                          visualise=False, feature_vector=feature_vector_flag)
        return hog_features
    
    
    
    
    
    

# In[4]:    

#testing HOG on test images

image=pathole_images_original[30]
image= cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
image_channel_0=image[:,:,0]
image_channel_1=image[:,:,1]
image_channel_2=image[:,:,2]

feature_0,hog_img_0=GetFeaturesFromHog(image_channel_0,9,2,16,visualise=True,feature_vector_flag=True)
feature_1,hog_img_1=GetFeaturesFromHog(image_channel_1,9,2,16,visualise=True,feature_vector_flag=True)
feature_2,hog_img_2=GetFeaturesFromHog(image_channel_2,9,2,16,visualise=True,feature_vector_flag=True)

f, axes= plt.subplots(1,4,figsize=(20,10))
axes[0].imshow(pathole_images_original[30])
axes[1].imshow(hog_img_0)
axes[2].imshow(hog_img_1)
axes[3].imshow(hog_img_2)


print("Feature Vector Length Returned is ",len(feature_0))
print("No of features that can be extracted from image ",len(hog_img_0.ravel()))   






image1=nopathole_images_original[45]
image1= cv2.cvtColor(image1, cv2.COLOR_RGB2YUV)
image_channel_0=image1[:,:,0]
image_channel_1=image1[:,:,1]
image_channel_2=image1[:,:,2]

feature_0,hog_img_0=GetFeaturesFromHog(image_channel_0,9,2,16,visualise=True,feature_vector_flag=True)
feature_1,hog_img_1=GetFeaturesFromHog(image_channel_1,9,2,16,visualise=True,feature_vector_flag=True)
feature_2,hog_img_2=GetFeaturesFromHog(image_channel_2,9,2,16,visualise=True,feature_vector_flag=True)

f, axes= plt.subplots(1,4,figsize=(20,10))
axes[0].imshow(nopathole_images_original[45])
axes[1].imshow(hog_img_0)
axes[2].imshow(hog_img_1)
axes[3].imshow(hog_img_2)


print("Feature Vector Length Returned is ",len(feature_0))
print("No of features that can be extracted from image ",len(hog_img_0.ravel()))   



# In[5]:  

#Convert Image Color Space. Note the colorspace parameter is like cv2.COLOR_RGB2YUV
def ConvertImageColorspace(image, colorspace):
    return cv2.cvtColor(image, colorspace) 





# Method to extract the features based on the choices as available in step 2

def ExtractFeatures(images,orientation,cellsPerBlock,pixelsPerCell, convertColorspace=False):
    featureList=[]
    imageList=[]
    for image in images:
        if(convertColorspace==True):
            image= cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        local_features_1=GetFeaturesFromHog(image[:,:,0],orientation,cellsPerBlock,pixelsPerCell, False, True)
        local_features_2=GetFeaturesFromHog(image[:,:,1],orientation,cellsPerBlock,pixelsPerCell, False, True)
        local_features_3=GetFeaturesFromHog(image[:,:,2],orientation,cellsPerBlock,pixelsPerCell, False, True)
        x=np.hstack((local_features_1,local_features_2,local_features_3))
        featureList.append(x)
    return featureList
    


# In[6]:  



#time
orientations=9
cellsPerBlock=2
pixelsPerBlock=16
convertColorSpace=True
patholeFeatures= ExtractFeatures(pathole_images_original,orientations,cellsPerBlock,pixelsPerBlock, convertColorSpace)
nopatholeFeatures= ExtractFeatures(nopathole_images_original,orientations,cellsPerBlock,pixelsPerBlock, convertColorSpace)




featuresList= np.vstack([patholeFeatures, nopatholeFeatures])
print("Shape of features list is ", featuresList.shape)
labelList= np.concatenate([np.ones(len(patholeFeatures)), np.zeros(len(nopatholeFeatures))])
print("Shape of label list is ", labelList.shape)



# train test split of data

from sklearn.model_selection import train_test_split

#X_train,  X_test,Y_train, Y_test = train_test_split(featuresList, labelList, test_size=0.2, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(featuresList, labelList, test_size=0.2, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)



# normalization and scaling

from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
scaler.fit(X_train)
X_train_scaled= scaler.transform(X_train)
X_test_scaled= scaler.transform(X_test)
X_valid_scaled= scaler.transform(X_val)


# Train a Linear SVM classifer
from sklearn.svm import SVC
#classifier1= SVC(kernel='rbf')
classifier1= SVC(kernel='linear')
#classifier1= SVC(kernel='poly')
#classifier1= SVC(kernel='sigmoid')


classifier1.fit(X_train,y_train)

print("Accuracy of SVM is  ", classifier1.score(X_test,y_test) )






predictvalid=classifier1.predict(X_val)


score1 = classifier1.score(X_test, y_test)
score2 = classifier1.score(X_val, y_val)
print("Accuracy of Test data=",score1,"Accuracy of Validation data=",score2)



# In[7]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


#Create a Gaussian Classifier
RFC=RandomForestClassifier(n_estimators=15)

#Train the model using the training sets y_pred=clf.predict(X_test)
RFC.fit(X_train,y_train)

y_pred=RFC.predict(X_test)

score3 = RFC.score(X_test, y_test)
score4 = RFC.score(X_val, y_val)
print("Accuracy of RandomForest Test data=",score3,"Accuracy of RandomForest Validation data=",score4)



# In[8]:

from sklearn.tree import DecisionTreeClassifier

DT=DecisionTreeClassifier(random_state=0)

#Train the model using the training sets y_pred=clf.predict(X_test)
DT.fit(X_train,y_train)

y_pred=DT.predict(X_test)

score5 = DT.score(X_test, y_test)
score6 = DT.score(X_val, y_val)
print("Accuracy of DecisionTree Test data=",score5,"Accuracy of DecisionTree Validation data=",score6)




# In[9]:
from sklearn.neighbors import KNeighborsClassifier


KNN= KNeighborsClassifier(n_neighbors=2)

#Train the model using the training sets y_pred=clf.predict(X_test)
KNN.fit(X_train,y_train)

y_pred=KNN.predict(X_test)

score7 = KNN.score(X_test, y_test)
score8 = KNN.score(X_val, y_val)
print("Accuracy of KNN Test data=",score7,"Accuracy of KNN Validation data=",score8)




# In[10]:


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    Recal=(TP)/(TP+FN) 
    Presicion=(TP)/(TP+FP)      
    return("TP=",TP, "FP=",FP, "TN=",TN, "FN=",FN,"recall=",Recal,"presicion=",Presicion)
    
    

probs=[]    
for p in range (y_val.shape[0]):
    if predictvalid[p]>0.5:
        probs.append(1)
    else:
        probs.append(0)
            
perf_measure(y_val,probs )




