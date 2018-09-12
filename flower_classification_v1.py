"""
    Keras Model for flower classification.
    
"""
from keras.utils import to_categorical
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import json



def main():
    """
    Assuming structre of data_dir as
    ../categories
    ../categories/category1
    ../categories/category2
    ../categories/category3
                
    """
    data_dir = 'C:/Users/Chiku/Downloads/flowers-recognition/flowers/' #provide path for raw data dir(catgories).
    save_dir = 'C:/Machine Learning/Work/test' #provide path for saving npz data.

    model_name = 'v3' #provide     
    
    save_np_data(data_dir,save_dir,model_name)    
    train(save_dir,model_name)
    test(save_dir,model_name)


def pre_processing(im_path,size=(128,128)):
    
    """
    :param im_path : path of image to transform for prediction.
    :return resized and normalised float array of image.
    
    This function will transform image for prediction and/or training 
    by reading, resizing image, converting to float type for normalisation.
    Normalisation map [0,255] to [0,1] and helps in early convergence.
    
    """
    
    im = cv2.imread(im_path)
    im = cv2.resize(im,size).astype('float')/255.0
    
    return im

def generate_arrays(data_dir,im_size=(128,128)):
    """
        Generates numpy array of images.
        
        :data_dir : path of data with folders of each category
        :im_size : resizing size for preprocessing image
        :returns data,labels numpy arrays and label_dict dictionary
    """
    data = []
    labels = []
    label_dict = {}
    
    print('Loading images...')

    
    i = 0
    """ Reading files from data_dir """
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir,folder)
        for file in os.listdir(folder_path):
            
            file_path = os.path.join(folder_path,file)
            
            
            """passing image file path for transformation for training."""
            
            im = pre_processing(file_path)
            
        
            """Appending image and labels to data and labels list respectively."""
    
            data.append(im)
            labels.append(i)
            
        """ Label_dict will map integer to flower(class) name.
            labels are given as int because of later transformation 
            of labels take integer list or array and because Neural Networks
            can make sense of numbers only.Labels in ASCII format won't make
            sense.
        """
        label_dict[i] = folder
        i += 1
        
    print(len(data))
    """Converting list of images and labels to numpy array """
    data = np.asarray(data)
    labels = np.asarray(labels)
        
    return data,labels,label_dict



def save_np_data(data_dir,save_dir,save_name,im_size=(128,128,3),
                 validation_fraction = 0.2,random_seed = 1):
   
    """
    :data_dir : path of raw data
    :save_dir : path to save npz data
    :save_name : name of data
    :im_size : for preprocessing
    :validation_fraction : split fraction of validation set, float <- (0,1.0)
    :random_seed : for reproducability with same seed.
    
    data is saved in train and test npz arrays along with labels 
    and class labels dictionary.
    
    """
    X,Y,Z = generate_arrays(data_dir)
    
    """ As labels are not related to each other but NN can take their 
    natural order as some sort of relationship which can mess up 
    with prediction so one hot encoding work as dummy variable by 
    replacing integer labels.
    
    integers labels are mapped to binary form
    e.g 4 will be represented as [0,0,0,0,1] """
    
    Y = to_categorical(Y)
    
    print('Spliting data...')
    """Split the data into train and test """   
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=validation_fraction
                                                     ,random_state = random_seed)
    
    
    
    print('Saving data..')
    save_path_train = os.path.join(save_dir,save_name+'_train')
    save_path_test = os.path.join(save_dir,save_name+'_test')
    
    
    
    np.savez(save_path_train,X_train=X_train,Y_train=Y_train,Z=Z)
    np.savez(save_path_test,X_test=X_test,Y_test = Y_test,Z=Z)
    


def load_np_data(data_path,set_='train'):
    """
    :data_path : absolute path of npz data 
    
    load npz data into arrays.
    """
    print('Loading data. . .')
    data = np.load(data_path)
    X = 'X_{}'.format(set_)
    Y = 'Y_{}'.format(set_)
    return data[X],data[Y],data['Z']
    
    
def train(data_dir,model_name,im_size=(128,128,3),dropout_fraction=0.2,
          epochs=10):
    """
    :data_dir : npz data directory
    :model_name : name of training data and model
    :im_size : size of image for input layer
    :dropout_fraction : fraction of weights to drop
    :epochs : number of full passes of training.
    
    define architechture and compile model.
    train model
    save model and json file
    
    """
    
    data_path = os.path.join(data_dir,model_name+'_train.npz')
    X,Y,label_dict = load_np_data(data_path)

    
    """ Define architechture of Keras Model.
    Intialise blank model object which can add layers in sequential form. """
    model = Sequential()
    
    """Adding conolution layers with 32 filters, kernel size of 3x3 with input
    of shape (128,128,3) and activation relu """
    
    model.add(Conv2D(32, (3,3),input_shape=im_size,activation='relu'))
    
    """ Max pooling layer to reduce size of input to next layer. """
    model.add(MaxPool2D(pool_size=(2,2)))
    
    """Adding conolution layers with 15 filters, kernel size of 3x3 with input
    of shape (128,128,3) and activation relu """
    
    model.add(Conv2D(filters=15, kernel_size=(3, 3), activation='relu'))
    
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    """Droping out 20% of weights randomly."""
    model.add(Dropout(dropout_fraction))
    
    """Converting n-Dimensional array to 1-D array for fully connected layers followed."""
    model.add(Flatten())
    
    """Adding dense layers with output of 128 with activation relu"""
    model.add(Dense(128,activation='relu'))
    
    """Adding dense layers with output of 5(number of classes) in this case
       Softmax activation will give probability of each class
       index of max probablity will become keys of label dictionary.
    """
    model.add(Dense(5,activation='softmax'))
    
    """ Compiling model with params of loss,optimizer and metrics for model evaluation """
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    """
        Model Training
        with ephochs(int) full passes
        with batch size of 128, weights will be updated after every 128
        images.
    """
    
    print("Training....")
    model.fit(X, Y, epochs=epochs, batch_size=128,verbose=1)
    
    
    model_path = os.path.join(data_dir,model_name+'.h5')
    """ Save the model """
    print("Saving model..")
    model.save(model_path)
    print('Done.') 
    
    """ Save json file for categories"""
    
    print("Saving json..")
    label_dict = label_dict.tolist()
    jsonfile = os.path.join(data_dir,model_name+'.json')
    
    with open(jsonfile, 'w') as f:
        json.dump(label_dict, f)
    
    
    

def test(data_dir,model_name,plot=True):
    """
    :data_dir : npz data directory 
    :model_name : test data and model name
    :plot : boolean value plot confusion matrix if true
    
    evaluates model and plot confusion matrix
    
    """
    data_path = os.path.join(data_dir,model_name+'_test.npz')
    
    X,Y,label_dict = load_np_data(data_path,'test')
    
    label_dict = label_dict.tolist()
    
    model_path = os.path.join(data_dir,model_name+'.h5')
    model = load_model(model_path)
    
    print(X.shape)

    # show the accuracy on the testing set
    print("Evaluating the testing set...")
    (loss, accuracy) = model.evaluate(X, Y,
                                      batch_size=128, verbose=2)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    if plot:
        pred_scores = model.predict(X,
                                    batch_size=128)
        pred_Y = np.argmax(pred_scores, axis=1)
        
        y_true = np.argmax(Y,axis=1)
        y_pred = pred_Y
        
        
        
        cm = confusion_matrix(y_true=y_true,
                                y_pred=y_pred,
                                labels= list(label_dict.keys()))
        return cm
    
    


def loadModel(model_dir,model_name):
    """
    :model_dir : directory of model
    :model_name : name of model to be loaded
    :returns model and label dictionary
    """
    
    model_path = os.path.join(model_dir,model_name)    
    model = load_model(model_path+'.h5')
    
    with open(model_path+'.json','r') as f:
        label_dict = json.load(f)
    
    return model,label_dict
    

def predictImage(im_path,model,label_dict):   
    """
    :im_path : path of image
    :model : model to predict
    :label_dict : int -> label
    
    prints label of image.
    """
    
    
    im = pre_processing(im_path)
    
    im = np.array([im])
    preds = model.predict(im)
    
    
    pred = label_dict[str(np.argmax(preds))]   
    print("The class of given image is {}.".format(pred))
    
    
if __name__ == '__main__':
    main()   
    
    
    
    