from keras.models import load_model
classifier = load_model('10-monkey-species/monkey.h5')

#Part 1 - Building Convolutional Neural Network

#Importing Libraries and Packages
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Intializing the CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (300, 300, 3), activation = "relu"))

#Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding a second convolution layer
classifier.add(Convolution2D(32, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding a third convolution layer
classifier.add(Convolution2D(64, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(activation="relu", units=256))
classifier.add(Dropout(0.5))
classifier.add(Dense(activation="softmax", units=10))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
       rescale=1./255,
       shear_range=0.2,
       zoom_range=0.2,
       horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
       '10-monkey-species/training',
       target_size=(300, 300),
       batch_size=32,
       class_mode='categorical')

test_set = test_datagen.flow_from_directory(
       '10-monkey-species/validation',
       target_size=(300, 300),
       batch_size=32,
       class_mode='categorical')

classifier.fit_generator(
       training_set,
       steps_per_epoch=(1097/32),
       epochs=25,  
       validation_data= test_set,
       validation_steps= (270/32))

classifier.save('monkey.h5')
print("Model is saving...")


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('10-monkey-species/validation/n2/n202.jpg',target_size=(300, 300) )
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

