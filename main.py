from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

## Model Loading
model=load_model("fashion_MNIST_classifier.h5")

## load Class label
class_label=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']



def prediction_function(path):
    img=image.load_img(path,target_size=(28,28),color_mode="grayscale")     # Load Image
    img_array=image.img_to_array(img)                                       # convert to image_array; (28,28,1)
    expanded=np.expand_dims(img_array,axis=0)                               # expand dimension; (1,28,28,1)

    predictions=model.predict(expanded)                                     # model prediction
    return class_label[np.argmax(predictions)]                              # prediction

