import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout2D, Activation, Lambda, Flatten, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import utils

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np






train_folder = 'D:/ML/lumgs_cancer_classification_ct_scan/Data/train'
test_folder = 'D:/ML/lumgs_cancer_classification_ct_scan/Data/test'
validate_folder = 'D:/ML/lumgs_cancer_classification_ct_scan/Data/valid'

normal_folder = '/normal'
adenocarcinoma_folder = '/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib'
large_cell_carcinoma_folder = '/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa'
squamous_cell_carcinoma_folder = '/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'




IMAGE_SIZE = (350, 350)

print("Reading training images from:", train_folder)
print("Reading validation images from:", validate_folder)

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 6

train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical'
)



OUTPUT_SIZE = 4

pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
pretrained_model.trainable = False

model = Sequential()
model.add(pretrained_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(OUTPUT_SIZE, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# /////////////////
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

def predict_image_class(img_path, model_path, class_labels, save_directory, image_size=(350, 350)):
    """
    Predict the class of an image using the trained model, display the image, and save the image with prediction.
    
    Parameters:
    - img_path (str): The path to the image file.
    - model_path (str): The path to the trained model file.
    - class_labels (list): A list of class labels.
    - save_directory (str): The directory to save the predicted image.
    - image_size (tuple): The target size to which the image will be resized. Default is (350, 350).

    Returns:
    - predicted_label (str): The predicted class label.
    - confidence_score (float): The confidence score for the prediction.
    """
    # Load the trained model
    model = load_model(model_path)
    
    # Define image preprocessing function
    def load_and_preprocess_image(img_path, target_size):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image to [0, 1]
        return img_array
    
    # Preprocess the image
    img = load_and_preprocess_image(img_path, image_size)
    
    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    confidence_score = predictions[0][predicted_class]  # Confidence of the predicted class

    # Map the predicted class to its label
    predicted_label = class_labels[predicted_class]

    # Print the predicted class and confidence score
    print(f"The image belongs to class: {predicted_label}")
    print(f"Confidence score for the prediction: {confidence_score:.4f}")
    
    # Load the image and create the plot
    plt.figure(figsize=(5, 5))
    plt.imshow(image.load_img(img_path, target_size=image_size))
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence_score:.2f})")
    plt.axis('off')

    # Save the image with the prediction
    os.makedirs(save_directory, exist_ok=True)  # Ensure the save directory exists
    save_path = os.path.join(save_directory, os.path.basename(img_path))  # Use the same filename as the input
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Predicted image saved at: {save_path}")
    
    plt.show()
    return predicted_label, confidence_score

# Example usage
img_path = 'Data/valid/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa/000111.png'
model_path = 'trained_lung_cancer_model.h5'
save_directory = 'predicted/'  # Directory to save the image
class_labels = list(train_generator.class_indices.keys())

# predict_image_class(img_path, model_path, class_labels, save_directory)
