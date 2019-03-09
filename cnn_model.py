from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization


def basicCNNbuild(batch_size=32,epoch=30,input_shape=(64,64,3)):
    batch_size = batch_size
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    return model


def load_images(path='train', validation_size=0.1, batch_size=32, target_size=(64,64)):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    validation_split=validation_size)  # set validation split

    train_generator = train_datagen.flow_from_directory(
        path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training', shuffle=True
        )  # set as training data

    validation_generator = train_datagen.flow_from_directory(
        path,  # same directory as training data
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical', shuffle=True,
        subset='validation')  # set as validation data

    return train_generator, validation_generator


def train_model(model, train_generator, validation_generator,batch_size=32):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=3000//batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=40 // batch_size)
    return history

def plot_loss(history):
# summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def show_model(model):
    plot_model(model, to_file='model.png')

if __name__ == "__main__":
    model = basicCNNbuild()
    (train_generator, validation_generator) = load_images()
    #print(train_generator.next()[0])
    history = train_model(model, train_generator, validation_generator)
    model.save('basicCNN_model.h5', overwrite=True)
    score = model.evaluate_generator(validation_generator,32)
    print("Loss: ", score[0], "Accuracy: ", score[1])
    plot_loss(history)