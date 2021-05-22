import cv2
import numpy as np

# specify the path of the pre-trained MobileNet SSD model
# use this model to detect the objects in a new image
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# the pre-trained model can detect a list of object classes,
# so we define those classes in a dictionary and a list
categories = {0: 'background', 1: 'airplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
              9: 'chair', 10: 'cow', 11: 'dining-table', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
              16: 'potted-plant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tv-monitor'}

# defined in list also
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "dining-table",  "dog", "horse", "motorbike", "person", "potted-plant", "sheep", "sofa", "train", "tv-monitor"]

image_counter = 0
dataset = "dataset/test_set/"
# find how to put every image in a variable
for image in dataset:
    image = cv2.imread('dataset/dog.jpg')  # change image name to check different results
    (h, w) = image.shape[:2]

    # MobileNet requires fixed dimensions for all input images, so first i resize
    # the image to 300x300 pixels and then normalize it
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)  # feeding the scaled image to the network
    detections = net.forward() # finally we will get the name of the detected object with confidence scores

    colors = np.random.uniform(255, 0, size=(len(categories), 3))  # select random colors for the bounding boxes

    # iterating over all the detection results and discard
    # any output whose confidence/probability is less than 0.2
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # getting the confidence score

        if confidence > 0.2:  # checking if the confidence is less than 0.2
            idx = int(detections[0, 0, i, 1])  # get the index of the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # locate the position of detected object in an image
            print("... class: ", classes[idx], ", confidence score: ", confidence)

            (startX, startY, endX, endY) = box.astype("int")  # get the coordinate of bounding box
            label = "{}: {:.2f}%".format(classes[idx], confidence * 100)  # set label and confidence score
            cv2.rectangle(image, (startX, startY), (endX, endY), colors[idx], 4)  # create a rectangular box around the object

            y = startY - 15 if startY - 15 > 15 else startY + 15  # set position of text which is written on bounding box
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)  # write label of the detected box

# displaying the input image with detected objects
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
