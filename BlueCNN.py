import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adagrad
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


checkpointPath = "CNNTraining1/cp-{epoch:04d}.ckpt"
checkpointDir = os.path.dirname(checkpointPath)

def Grayscale(num = None):
    alive_location = 'Singles/Alive'
    dead_location = 'Singles/Dead'
    test_location = 'Test'
    
    folder = os.listdir(alive_location)
    alive_length = len(folder)
    
    folder = os.listdir(dead_location)
    dead_length = len(folder)
    
    folder = os.listdir(test_location)
    test_length = len(folder)
    
    if num == 1:
        for i in range(alive_length):
            name = str(1) + ' ' + '(' + str(i + 1) + ')'
            picture = image.load_img(alive_location + '/' + name + '.jpg', target_size = (150,150))
            picture = np.array(picture)
            picture = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
        
            new_name = str(i + 1) + '.jpg'
            new_name = cv2.imwrite(new_name, picture)
        
    if num == 2:
        for i in range(dead_length):
            name = str(1) + ' ' + '(' + str(i + 1) + ')'
            picture = image.load_img(dead_location + '/' + name + '.jpg', target_size = (150,150))
            picture = np.array(picture)
            picture = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
        
            new_name = str(i + 1) + '.jpg'
            new_name = cv2.imwrite(new_name, picture)
    
    if num == 3:    
        for i in range(test_length):
            name = str(1) + ' ' + '(' + str(i + 1) + ')'
            picture = image.load_img(dead_location + '/' + name + '.jpg', target_size = (150,150))
            picture = np.array(picture)
            picture = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
        
            new_name = str(i + 1) + '.jpg'
            new_name = cv2.imwrite(new_name, picture)
    
def Train():
    checkpointPath = "CNNTraining1/cp-{epoch:04d}.ckpt"
    currentTrainingDir = "Singles"

    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                   input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(3, 3),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(3, 3),
            #tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            #tf.keras.layers.MaxPooling2D(1, 1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')
            ])
     
    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy',
                  metrics=['acc'])
     
    scaledDatagen = ImageDataGenerator(rescale = 1.0/255.)
     
    trainGenerator = scaledDatagen.flow_from_directory(currentTrainingDir, 
                                                       batch_size = 25,
                                                       target_size=(150,150))
     
    cpCallback = tf.keras.callbacks.ModelCheckpoint(checkpointPath,
                                                    save_weights_only = True,
                                                    verbose = 1)
     
    model.summary()
    history = model.fit_generator(trainGenerator, epochs = 20,
                                  callbacks = [cpCallback])
    return history, model
    
def Train_CNN():
    checkpointPath = "CNNTraining2/cp-{epoch:04d}.ckpt"
    currentTrainingDir = "New Set"
    
    layers = [
            tf.keras.layers.Conv2D(32, (3,3), activation = 'relu',
                                   input_shape = (150,150,1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')]
    model = tf.keras.Sequential(layers)
    
    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy',
                  metrics=['acc'])
    
    train_images = []
    train_label = []
    
    alive_images_num = os.listdir(currentTrainingDir + '/Alive Gray')
    dead_images_num = os.listdir(currentTrainingDir + '/Dead Gray')
    
    for num in alive_images_num:
        train_label.append(1)
        
    for num in dead_images_num:
        train_label.append(0)
    
    for pic in alive_images_num:
        train_images.append(load_image((currentTrainingDir + '/Alive Gray/') + pic))
        
    for pic in dead_images_num:
        train_images.append(load_image((currentTrainingDir + '/Dead Gray/') + pic))
    
    #print(train_images)
    #print(train_label.shape)
    
    train_images = np.array(train_images)
    train_label = np.array(train_label)
    
    #print(train_label.shape)
    
    #train_label_binary = []
    train_label_binary = to_categorical(train_label)
    
    train_images = np.expand_dims(train_images, axis = -1)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpointPath,
                                                    save_weights_only = True,
                                                    verbose = 1)
    
    model.fit(train_images, train_label_binary, epochs = 5, batch_size = 100,
              callbacks = [cp_callback])
    return model
    
def LoadModel():
     model = tf.keras.models.Sequential([
               tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                      input_shape=(150, 150, 1)),
               tf.keras.layers.MaxPooling2D(2, 2),
               tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
               tf.keras.layers.MaxPooling2D(2, 2),
               tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
               tf.keras.layers.MaxPooling2D(2, 2),
               tf.keras.layers.Flatten(),
               tf.keras.layers.Dense(512, activation='relu'),
               tf.keras.layers.Dense(2, activation='sigmoid')
               ])
     latest = tf.train.latest_checkpoint(checkpointDir)
     model.load_weights(latest)
     
     return model
 
def load_image(file_path):
    temp = []
    temp = np.array(temp)
    temp = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    return temp

def Graph(history):
     acc=history.history['acc']
     loss=history.history['loss']
     
     epochs=range(len(acc)) 
     
     plt.plot(epochs, acc, 'r', "Training Accuracy")
     plt.title('Training and validation accuracy')
     plt.figure()
     
     plt.plot(epochs, loss, 'r', "Training Loss")
     plt.figure()
     
def Visualizer(img, x):
     successive_outputs = [layer.output for layer in model.layers[1:]]
     
     visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
     
     successive_feature_maps = visualization_model.predict(img)
     
     layer_names = [layer.name for layer in model.layers]

     for layer_name, feature_map in zip(layer_names, successive_feature_maps):
          if len(feature_map.shape) == 4:
               n_features = feature_map.shape[-1]
               size = feature_map.shape[1]
               
               display_grid = np.zeros((size,size * n_features))
               
               for i in range(n_features):
                    img = feature_map[0, :, :, i]
                    img -= img.mean()
                    img = img / img.std()
                    img *= 64
                    img += 128
                    img = np.clip(img, 0, 255).astype('uint8')
                    display_grid[:, i * size : (i + 1) * size] = img
               
               scale = 20. / n_features
               plt.figure(figsize=(scale * n_features, scale))
               plt.title (layer_name + ' ' + str(x + 1))
               plt.grid (False)
               plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')
               
def Predict(model):
     predictPath = 'Test'
     
     #decryption = [[1,0],[0,1]]
     folder = os.listdir(predictPath)
     numOfPic = len(folder)
     
     for x in range(numOfPic):
          picture = (predictPath + '/' + str(x + 1) + '.jpg')
          img = cv2.imread(picture, cv2.IMREAD_GRAYSCALE)
          img = np.array(img)
          #print(img.shape)
          img = np.expand_dims(img, axis = 0)
          img = np.expand_dims(img, axis = -1)
     
          images = np.vstack([img])
          #print(images.shape)
          classes = model.predict(images, batch_size=10)       
          
          Visualizer(img,x)
          print (x)
          print(classes)
          
          if np.argmax(classes) == 1:
               print('Alive')
          else:
              print('Dead')

def Image_Gen_Predict(model):
    decryption = [[1,0,0],[0,1,0],[0,0,1]]
     
    predictPath = 'Test'
     
    folder = os.listdir(predictPath)
    numOfPic = len(folder)
     
    for x in range(numOfPic):
         picture = image.load_img(predictPath + '/' + str(x + 1) + '.jpg', target_size=(150,150))
         img = image.img_to_array(picture)
         img = np.expand_dims(img, axis = 0)
     
         images = np.vstack([img])
         classes = model.predict(images, batch_size=10)       
          
         Visualizer(img,x)
         print (x)
          
         if np.argmax(classes) == 0:
              print('Alive')
         elif np.argmax(classes) == 1:
              print('Dead')
               
#Grayscale(2)
history, model = Train()
#model = LoadModel()
#model = Train_CNN()
#Predict(model)
Image_Gen_Predict(model)
