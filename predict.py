from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
model = load_model('basicCNN_model.h5')
x = cv2.imread(
    'tt.jpg')[..., ::-1]

x = cv2.resize(x,(64,64))
x = x[np.newaxis, ...]
x = x/255.

pred = model.predict(x)
print(pred)