import cv2

img_file = 'car2.png'
classifier_file = 'cars.xml'

img = cv2.imread(img_file)
video = cv2.VideoCapture('car video.mp4')

car_tracker = cv2.CascadeClassifier('cars.xml')
pd_tracker = cv2.CascadeClassifier('pedestrian.xml')

while True:
    
    (read_successful, frame) = video.read()

    if read_successful:
        gf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(gf)
    ped = pd_tracker.detectMultiScale(gf)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x+1,y+2), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

    for (x,y,w,h) in ped:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('car detector', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break


print('code completed')