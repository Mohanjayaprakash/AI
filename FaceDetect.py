import cv2 as cv
import sys

#print(cv2.__version__)

# Reading the image
src=sys.argv[1]
#src=Image.open(recent.jpeg)
classiImgr=cv.CascadeClassifier('haarcascade_frontalface_default.xml') #classifier for detecting faces

#Reading the images and converting to grayscale
img=cv.imread(src)
grayImg=cv.cvtColor(img,cv.COLOR_RGB2GRAY)

#Detecting the faces in the imag.....Syntax: detectMultiScale(image, rejectLevels, levelWeights)
faceDetect=classiImgr.detectMultiScale(
 grayImg,
 scaleFactor=1.5, #As the images get more burred the cale Factor increases
 minNeighbors=5,  #
 minSize=(30, 30),
 flags = cv.CV_HAAR_SCALE_IMAGE # New Syntax in cv3
)

#print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces .... the for loop iterates over to detect rectangles where it thinks images are there
for (x, y, w, h) in faceDetect: # Any image has 4 parameters 
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
      
cv.imshow("Detected faces",img) # Display the image with the detected faces
cv.waitKey(0) # the output window waits until we close else without this function it closes immediatly after showing the result


