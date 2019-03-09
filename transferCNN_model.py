from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization
from keras import applications
from keras.layers import Input
from cnn_model import *
# Building a network with bottom layers taken from VGG
vgg_model = applications.VGG16(weights='imagenet',
                               include_top=False)

for layer in vgg_model.layers:
   layer.trainable = False
input_tensor = Input(shape=(64,64,3), name='image_input')

output_vgg16_conv = vgg_model(input_tensor)

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(1000, activation='relu', name='fc1')(x)
x = Dropout(0.5,name='dropout')(x)
x = Dense(1000, activation='relu', name='fc2')(x)
x = BatchNormalization(name = 'normal')(x)
x = Dense(2, activation='softmax', name='predictions')(x)
new_model = Model(input=input_tensor, output=x)
new_model.summary()
# vgg_model.summary()


new_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

(train_generator, validation_generator) = load_images(target_size=(64,64))
history = train_model(new_model,train_generator,validation_generator,batch_size=32)
new_model.save('transferCNN_model.h5', overwrite=True)
score = new_model.evaluate_generator(validation_generator, 32)
print("Loss: ", score[0], "Accuracy: ", score[1])
