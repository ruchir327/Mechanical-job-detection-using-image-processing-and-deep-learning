import numpy as np
import json
import cv2
import keras

jsonFileName='detect.json'
model=keras.models.model_from_json(json.load(open(jsonFileName,'r')))
model.load_weights('detect.hdf5')
print("done loading weights of trained model")

labels = ['1','2','3','4','5','6','7','unknown obj']

cropped = cv2.imread('1.jpg')
frame = cropped
cv2.imshow('input',cropped)

cropped = cv2.resize(cropped,(200,200))
cropped= cropped/255
cropped = np.reshape(cropped,[1,200,200,3])
classes = model.predict(cropped)
output = np.argmax(classes)
print(labels[output])


font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(frame, labels[output], (30,30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imwrite('output.jpg',frame)
cv2.imshow('input',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()