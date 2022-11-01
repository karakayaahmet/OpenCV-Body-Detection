import cv2

cap = cv2.VideoCapture("body.mp4")

body_cascade = cv2.CascadeClassifier("fullbody.xml")

while True:
    ret, frame = cap.read()

    if ret is False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies = body_cascade.detectMultiScale(gray, 1.2, 7)

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(50) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()