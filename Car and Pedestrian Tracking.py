import cv2

# Our Image
img_file='car_image.jpg'
# video = cv2.VideoCapture('tesla.mp4')
video = cv2.VideoCapture('pedestrian.mp4')

#car classifier
classifier_file = 'car_detector.xml' 


#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)
pedestrian_tracker = cv2.CascadeClassifier('pedestrian.xml')



#Run forever
while True:

    #Read the current frame
    (read_successful, frame) = video.read()

    #Safe Coding.
    if read_successful:
        #Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrian = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # #Draw rectangle around the cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for (x,y,w,h) in pedestrian:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)


    cv2.imshow('Detector', frame)

    key = cv2.waitKey(1)

    #Stop if Q key is pressed
    if key==81 or key==113:
        break

#Release the videoCaptureobject
video.release()

print("code Completed")

# #create opencv image
# img = cv2.imread(img_file)

# #convert to grayscale (needed for haar cascade)
# black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #detect cars
# cars = car_tracker.detectMultiScale(black_n_white)

# #Draw rectangle around the cars
# for (x,y,w,h) in cars:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)



# #Display the image with the faces spotted
# cv2.imshow('car detector', img)

# #Dont autoclose (Wait in the code and listen for a key press)
# cv2.waitKey()

