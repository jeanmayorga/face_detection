import cv2

face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_alt.xml')

def detect_faces(frame, is_video = False):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if is_video:
        frame_gray = cv2.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray)

    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
    return frame

def display_frame(frame):
    cv2.imshow('Face detection', frame)

def get_web_cam():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, frame = cap.read()

        frame_with_faces = detect_faces(frame, True)
        display_frame(frame_with_faces)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def get_image(image_url):
    frame = cv2.imread(image_url)
    
    frame_with_faces = detect_faces(frame)
    display_frame(frame_with_faces)

    cv2.waitKey(0)

## from images
# get_image('resources/person.jpg')
get_image('resources/friends.jpg')

# from web cam
# get_web_cam()