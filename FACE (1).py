#!/usr/bin/env python
# coding: utf-8

# In[27]:


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model #keras framework is used to load our deep learning  model
import cv2             #numpy and opencv cv2 libraries is to work with live webcam and images           
import numpy as np
import tkinter                   #tkinter is used  to create a warning pop up window in order to the user.if he or she is not wearing a mask
from tkinter import messagebox
import smtplib            #smtplib is being used to define  an smtpclient session object  that is used to send  an alert email if a person is found not wearing a mask
from keras.preprocessing import image


# In[28]:


get_ipython().system('pip install opencv-python #installing opencv ')


# In[29]:


# Initialize Tkinter
root = tkinter.Tk()#Initializing  Tkinter in order to create tk root which is a window  with a little bar and other decoration provided by the window  manager.
root.withdraw()#if we don't use this  function the it will cause the app to create a empty root window always which we  don't want.


# In[30]:


import tensorflow.keras as keras


# In[31]:


import os


# In[32]:


#Load trained deep learning model
model = load_model('model-090.model')#load model method  to load our deep learning model


# In[33]:


#Classifier to detect face
#we are using opencv haarcascade frontface classifier to detect the face in  the  videoframe so that first we will locate the vision of interest
#which in our case in human face ,Once we locate face,we can use our deep learning model loaded over to predict the person is wearing a mask or not
face_det_classifier= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[34]:


print(1)


# In[35]:


# Capture Video
vid_source=cv2.VideoCapture(0)#/In this line we are trying   to capture the live video feed by making use of opencv videocapture method this will  return video from the first web cam from the computer,Please ensurer that the argument inside is 0.


# In[36]:


#Here 0 and 1 are keys and mask and no mask are the values will be display in the rectangle  shape that we are going to draw around faces.
# Dictionaries containing details of Wearing Mask and Color of rectangle around face. If wearing mask then color would be 
# green and if not wearing mask then color of rectangle around face would be red
text_dict={0:'Mask ON',1:'No Mask'}
rect_color_dict={0:(0,255,0),1:(0,0,255)}
#rect_color_dict=it holds the colour of rectangle around the face,0 and 1 are keys [0,255,0]=green colour detection..i.e=MASK is weared and [0,0,255]=Red Colour detection..i.e=NO MASK is weared{Here(0,255,0)and(0,0,255)->represent the colour on BGRB}


# In[37]:


print(2)


# In[ ]:





# In[38]:


print(3)


# In[39]:


#while loop will continously detect the camera feed
while(True):

    ret,img=vid_source.read()# Here ret is boolean value  that tells whether or not  any frame  is return from the video feed.
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#it convert image  file into grayscale,For this we are using cv2 color function;here we are passing each frame along with a parameter(color.BGR2GRAY)
    faces=face_det_classifier.detectMultiScale(gray,1.3,5)#face_det_classifier method called as detect multiscale it takes several i/p parameter 1./p grayscale image 2.scale factor which  detect the parameter specifying how much the img size is reduce at  each img  scale by setting it to  1.3 which means that we are reducing the img by 30%.
                                                          #3.Means neighbour that depict  how many neighbour each candidate rectangle should have to return,this means that we have multiple faces in the same region then it may draw multiple rectangle  there,so we threw this parameter are telling that it should consider it as  one face.
    for (x,y,w,h) in faces:#Here x,y,w,h are coordinates in order to draw the rectangle,w=wide,h=height
        face_img = gray[y:y+w,x:x+w]
        resized_img = cv2.resize(face_img,(100,100))#here it is croping the image
        normalized_img = resized_img/255.0
        reshaped_img = np.reshape(normalized_img,(1,100,100,1))
        result=model.predict(reshaped_img)#here we are predicting the facemask used to predict function associated with the deep learning model.
        label=np.argmax(result,axis=1)[0]
        cv2.rectangle(img,(x,y),(x+w,y+h),rect_color_dict[label],2)#rectangle colour
        cv2.rectangle(img,(x,y-40),(x+w,y),rect_color_dict[label],-1)#rectangle color
        cv2.putText(img, text_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2) #here the text in the rectangle i.e MASK and i.e NO MASK
        if (label == 1):#if the label is equal to 1 than you are not wearing a mask and 0 means Wearing a mask.
            messagebox.showwarning("Warning","Access Denied. Please wear a Face Mask")
                        # Send an email to the administrator if access denied/user not wearing face maskmessage
            mail = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            mail.login("sharmasauravbhai781@gmail.com", "dibrugarh")
            mail.sendmail("sharmasauravbhai781@gmail.com","callmesaurav861@gmail.com","One Visitor violated Face Mask Policy. See in the camera to recognize user. A Person has been detected without a face mask in the University Campus. Please Alert the authorities."
)
            mail.quit()
        else:
            pass
            break

    cv2.imshow('LIVE Video Feed',img)#the camera frame with the name live video feed.
    key=cv2.waitKey(1)

    if(key==27):#if we are pressing the escape key it will quit
        break

cv2.destroyAllWindows()
source.release()

    


# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:









