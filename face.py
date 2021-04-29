from imutils import paths
import numpy as np
import cv2
import os

def detect_faces(net, image, minConfidence=0.5):
	
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	
	net.setInput(blob)
	detections = net.forward()
	boxes = []

	
	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]
		#지정한 확률보다 낮으면 무시하고 진행
		if confidence > minConfidence:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			boxes.append((startX, startY, endX, endY))

	# return the face detection bounding boxes
	return boxes

def load_face_dataset(inputPath, net, minConfidence=0.5,minSamples=5):
	imagePaths = list(paths.list_images(inputPath))
	names = [p.split(os.path.sep)[-2] for p in imagePaths]
	(names, counts) = np.unique(names, return_counts=True)
	names = names.tolist()

	
	# labels
	faces = []
	labels = []

	# loop over the image paths
	for imagePath in imagePaths:
		image = cv2.imread(imagePath)
		name = imagePath.split(os.path.sep)[-2]

		#안쓸예정
		if counts[names.index(name)] < minSamples:
			continue

		# perform face detection
		boxes = detect_faces(net, image, minConfidence)

		
		for (startX, startY, endX, endY) in boxes:
			# extract the face ROI, resize it, and convert it to
			# grayscale
			# skleran에 있는 data format도 62 x 47 이어서 PCA의 component분석결과를 적용시키기 위해 resize
			faceROI = image[startY:endY, startX:endX]
			faceROI = cv2.resize(faceROI, (47, 62))
			faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

			# update our faces and labels lists
			faces.append(faceROI)
			labels.append(name)

	# convert our faces and labels lists to numpy arrays
	faces = np.array(faces)
	labels = np.array(labels)

	# return a 2-tuple of the faces and labels
	return (faces, labels)