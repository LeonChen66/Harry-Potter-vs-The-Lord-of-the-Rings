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
from keras.optimizers import Adam
# Building a network with bottom layers taken from VGG
vgg_model = applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(64, 64, 3))

for layer in vgg_model.layers[:10]:
   layer.trainable = False

x = vgg_model.output
x = Flatten(name='flatten')(x)
x = Dense(1024, activation='relu', name='fc1')(x)
x = Dropout(0.5,name='dropout')(x)
x = Dense(1024, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
prediction = Dense(2, activation='softmax', name='predictions')(x)
new_model = Model(input=vgg_model.input, output=prediction)
new_model.summary()


new_model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])

(train_generator, validation_generator) = load_images(target_size=(64,64))
history = train_model(new_model,train_generator,validation_generator,batch_size=32,epoch=100)
new_model.save('transferCNN_model.h5', overwrite=True)
score = new_model.evaluate_generator(validation_generator, 32)
print("Loss: ", score[0], "Accuracy: ", score[1])
plot_model(history)

