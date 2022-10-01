import numpy as np
import cv2
import torch

# DNN Model Initialization
classes = ["Axe"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device:", device)

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path = "Training Yolov5 model/best.pt",
    force_reload=True
)

def score_frame(frame):
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord

def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.3:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, (classes[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame

circles = np.zeros((4, 2), int)
counter = 0
def click_button(event, x, y, flag, params):
    global counter
    if (event == cv2.EVENT_LBUTTONDOWN):
        circles[counter] = x,y
        counter += 1

cap = cv2.VideoCapture("Testing Data/tourney64-long 2.mp4")
cv2.namedWindow("Input")
cv2.setMouseCallback("Input", click_button)

while cap.isOpened:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    for i, j in circles:
        cv2.circle(frame, (i, j), 5, (255, 255, 0), -1)

    if counter == 4:
        w, h = 416, 416
        pts1 = np.float32([circles[0][0:], circles[1][0:], circles[2][0:], circles[3][0:]])
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        frameOutput = cv2.warpPerspective(frame, matrix, (w, h))

        # Using DNN model
        results = score_frame(frameOutput)
        frameOutput = plot_boxes(results=results, frame=frameOutput)

        cv2.imshow("Output", frameOutput)

    cv2.imshow("Input", frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()