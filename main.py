import cv2
import numpy as np
from Neural_net import NeuralNet

net = NeuralNet()

# creating a 480 x 410 pixels canvas
canvas = np.ones((480, 410), dtype="uint8") * 255
# 400 x 400 pixels point of interest on which digits will be drawn
canvas[40:400, 10:400] = 0


start_point = None
end_point = None
is_drawing = False


def draw_line(img, start_at, end_at):
    cv2.line(img, start_at, end_at, 255, 20)


def on_mouse_events(event, x, y, flags,params):
    global start_point
    global end_point
    global canvas
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        if is_drawing:
            start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            end_point = (x, y)
            draw_line(canvas, start_point, end_point)
            start_point = end_point
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False


cv2.namedWindow("AI Project - Digit Recognizer")
cv2.setMouseCallback("AI Project - Digit Recognizer", on_mouse_events)

font = cv2.FONT_ITALIC
cv2.putText(canvas, 'INPUT', (150, 30), font, 1, (0, 0, 0))
cv2.putText(canvas, 'c - clean canvas   p - predict   t - train', (10, 445), font, 0.55, (0, 0, 0))


while True:
    cv2.imshow("AI Project - Digit Recognizer", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    elif key == ord('t'):
        net.training()
    elif key == ord('i'):
        net.info()
    elif key == ord('c'):
        canvas[40:400, 10:400] = 0
        print("Canvas cleaned")
    elif key == ord('p'):
        image = canvas[40:400, 10:400]
        result = net.predict(image)
        print("My prediction is: ", result)

cv2.destroyAllWindows()
