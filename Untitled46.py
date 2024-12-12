#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip show opencv-python


# In[5]:


pip install opencv-python


# In[3]:


import cv2
print(cv2.__version__)


# In[4]:


import os


# In[5]:


import os
if not os.path.exists("Selfie_Images"):     
 os.mkdir("Selfie_Images")


# In[6]:


import cv2
camera =cv2.VideoCapture(0)


# In[7]:


#Step 4: Loading Face Detection Model
import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[ ]:


#Step 5: Capturing Selfies
import cv2
import os
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not access the camera.")
else:
 selfie_count = 0
while selfie_count < 5:
 ret, frame = camera.read()     
if not ret:         
    break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for i, (x, y, w, h) in enumerate(faces):         
    face = frame[y:y+h, x:x+w]
    face_folder = f"Selfie_Images/Face_{i+1}"
if not os.path.exists(face_folder):             
        os.mkdir(face_folder) 
        face_image_path = os.path.join(face_folder, f"selfie_{selfie_count+1}.jpg")
        cv2.imwrite(face_image_path, face)
cv2.imshow("Webcam - Press 'q' to Quit", frame)
key = cv2.waitKey(1000)  
selfie_count += 1
if key == ord('q'):         
    break


# In[ ]:


#Step 6: Releasing Resources
import cv2
camera = cv2.VideoCapture(0)
camera.release() 
cv2.destroyAllWindows()


# In[ ]:


# step7:Edge Detection
import cv2

Selfie_Images="C:\python11\image.jpg"
# Load an image from file (Replace "your_image.jpg" with your image file path)
frame = cv2.imread(Selfie_Images)


# Check if the image was loaded successfully
if frame is None:
    print("Error: Image not found.")
else:
 edge_image = cv2.Canny(frame, 100, 200)
cv2.imshow("Original Image", frame) 
cv2.imshow("Edge Detected Image", edge_image) 
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[ ]:


# step8:Image Sharpening 
import numpy as np
import cv2
Selfie_Images="C:\python11\image.jpg"
frame = cv2.imread(Selfie_Images)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
sharpened_image = cv2.filter2D(frame, -1, kernel)
# Display original vs sharpened image 
cv2.imshow("Original Image", frame)
cv2.imshow("Sharpened Image", sharpened_image) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
import numpy as np
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
sharpened_image = cv2.filter2D(frame, -1, kernel)
# Display original vs sharpened image 
cv2.imshow("Original Image", frame)
cv2.imshow("Sharpened Image", sharpened_image)
cv2.waitKey(0) 
cv2.destroyAllWindows()




# In[ ]:


#blur image
import cv2

# Load an image (replace "your_image.jpg" with the path to your image file)
frame = cv2.imread("C:\python11\image.jpg")

# Check if the image was loaded successfully
if frame is None:
    print("Error: Image not found.")
else:
    # Step 9: Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(frame, (5, 5), 0)

    # Display original vs blurred image
    cv2.imshow("Original Image", frame)
    cv2.imshow("Blurred Image", blurred_image)

    # Wait for a key press to close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


# In[ ]:


# step10:Image Resize
import cv2
frame = cv2.imread("C:\python11\image.jpg")
resized_image = cv2.resize(frame, None, fx=0.5, fy=0.5)
# Display original vs resized image 
cv2.imshow("Original Image", frame) 
cv2.imshow("Resized Image", resized_image) 
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#step11: Image Rotation
import cv2
frame = cv2.imread("C:\python11\image.jpg")
(h, w) = frame.shape[:2] 
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0) 
rotated_image = cv2.warpAffine(frame, rotation_matrix, (w, h))
# Display original vs rotated image 
cv2.imshow("Original Image", frame) 
cv2.imshow("Rotated Image", rotated_image) 
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[ ]:


#Step 12: Image Augmentation
# Image Augmentation
import cv2
frame = cv2.imread("C:\python11\image.jpg")
flipped_image = cv2.flip(frame, 1)  # Horizontal flip
(h, w) = flipped_image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), -30, 1.0) 
augmented_image = cv2.warpAffine(flipped_image, rotation_matrix, (w, h))
resized_augmented = cv2.resize(augmented_image, (200, 200))
# Display original vs augmented image cv2.imshow("Original Image", frame)
cv2.imshow("Augmented Image", resized_augmented)
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[ ]:


#step13:Image Cropping
# Image Cropping
import cv2
frame = cv2.imread('C:\python11\image.jpg')
cropped_image = frame[50:200, 100:300]
# Display original vs cropped image 

cv2.imshow("Original Image", frame) 
cv2.imshow("Cropped Image", cropped_image) 
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[ ]:


#step14:Step 14: Convert RGB to Black & White and Negative
import cv2
# Convert Image to Black & White
frame = cv2.imread("C:\python11\image.jpg")
bw_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Create a Negative of the Image 
negative_image = 255 - frame
# Display original vs black & white vs negative image 
cv2.imshow("Original Image",frame) 
cv2.imshow("Black & White Image",bw_image) 
cv2.imshow("Negative Image",negative_image) 
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[ ]:


# Face Detection
import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
frame = cv2.imread('C:\python11\image.jpg')
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 
for (x, y, w, h) in faces:     
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow("Original Image", gray_frame)
cv2.imshow("Face Detected Image", frame)
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[1]:


#step16:identify facial features
import cv2
frame = cv2.imread('C:\python11\image.jpg')
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(gray_frame, 1.1, 10) 
for (ex, ey, ew, eh) in eyes:     
 cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
cv2.imshow("Original Image", gray_frame) 
cv2.imshow("Eyes Detected Image", frame) 
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




