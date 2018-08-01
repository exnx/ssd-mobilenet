# USAGE
# python real_time_object_detection_single_image_grid.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.40,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
# print("[INFO] starting video stream...")
# vs = VideoStream(src=1).start()
# time.sleep(2.0)
# fps = FPS().start()

# loop over the frames from the video stream
# while True:
# grab the frame from the threaded video stream and resize it
# to have a maximum width of 400 pixels



orig = cv2.imread("0020.jpg")
# frame = imutils.resize(frame, width=400)

f11 = orig[0:360,0:480]
f12 = orig[0:360,480:960]
f13 = orig[0:360,960:1440]
f14 = orig[0:360,1440:1920]

f21 = orig[360:720,0:480]
f22 = orig[360:720,480:960]
f23 = orig[360:720,960:1440]
f24 = orig[360:720,1440:1920]

f31 = orig[720:1080,0:480]
f32 = orig[720:1080,480:960]
f33 = orig[720:1080,960:1440]
f34 = orig[720:1080,1440:1920]

# cv2.imshow("Frame", f21)
# key = cv2.waitKey(0) & 0xFF
# exit()

# f11_off = (0,0)
# f12_off = (0,480)
# f13_off = (0,960)
# f14_off = (0,1440)
#
# f21_off = (360,0)
# f22_off = (360,480)
# f23_off = (360,960)
# f24_off = (360,1440)
#
# f31_off = (720,0)
# f32_off = (720,480)
# f33_off = (720,960)
# f34_off = (720,1440)

f11_off = (0,0)
f12_off = (480,0)
f13_off = (960,0)
f14_off = (1440,0)

f21_off = (0,360)
f22_off = (480,360)
f23_off = (960,360)
f24_off = (1440,360)

f31_off = (0,720)
f32_off = (480,720)
f33_off = (960,720)
f34_off = (1440,720)

grid = [f11,f12,f13,f14,f21,f22,f23,f24,f31,f32,f33,f34]
offset = [f11_off,f12_off,f13_off,f14_off,f21_off,f22_off,f23_off,f24_off,f31_off,f32_off,f33_off,f34_off]

# testing
# grid = [f33]
# offset = [f33_off]

# for off in offset:
#     cv2.circle(orig, off, 4, (20, 0, 255), -1)
#
# # show the output frame
# cv2.imshow("Frame", orig)
# key = cv2.waitKey(0) & 0xFF
# exit()

frame_boxes = []  # a list of tuples, each tuple have 4 x/y values.

for cell_num,cell in enumerate(grid):

    # grab the frame dimensions and convert it to a blob
    (h, w) = cell.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(cell, (300, 300)),
    	0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # # show the output frame
    # cv2.imshow("Frame", cell)
    # key = cv2.waitKey(0) & 0xFF
    # exit()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            if idx == 15:

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)

                cv2.rectangle(cell, (startX, startY), (endX, endY),
                    (255,0,0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(cell, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # show the output frame
                cv2.imshow("Frame", cell)
                key = cv2.waitKey(0) & 0xFF

            # apply offset according which cell detection found in
            startX += offset[cell_num][0]
            startY += offset[cell_num][1]
            endX += offset[cell_num][0]
            endY += offset[cell_num][1]
            detection_data = [startX,startY,endX,endY,idx,confidence]

            frame_boxes.append(detection_data)

for elem, detection_data in enumerate(frame_boxes):

    startX,startY,endX,endY,idx,confidence = detection_data

    # only show person detections
    if idx == 15:

        # draw the prediction on the frame
        # label = "{}: {:.2f}%".format(CLASSES[idx],
        #     confidence * 100)
        #
        # cv2.rectangle(orig, (startX, startY), (endX, endY),
        #     (255,0,0), 2)
        # y = startY - 15 if startY - 15 > 15 else startY + 15
        # cv2.putText(orig, label, (startX, y),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        print('bounding box for object {}:'.format(elem))
        print(startX, startY, endX, endY)

# show the output frame
cv2.imshow("Frame", orig)
key = cv2.waitKey(0) & 0xFF
#
# # if the `q` key was pressed, break from the loop
# if key == ord("q"):
# 	break

# # update the FPS counter
# fps.update()
#
# # stop the timer and display FPS information
# fps.stop()
# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
# cv2.destroyAllWindows()
# vs.stop()
