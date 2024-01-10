import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from tensorflow.keras.callbacks import Callback
from fastai.vision.all import *

class StreamlitCallback(Callback):
    def on_train_begin(self, logs=None):
        self.epoch_data = {'Epoch': [], 'Train Loss': [], 'Train Accuracy': [], 'Validation Loss': [], 'Validation Accuracy': []}
        self.table_placeholder = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_data['Epoch'].append(epoch)
        self.epoch_data['Train Loss'].append(logs['loss'])
        self.epoch_data['Train Accuracy'].append(logs['accuracy'])
        self.epoch_data['Validation Loss'].append(logs['val_loss'])
        self.epoch_data['Validation Accuracy'].append(logs['val_accuracy'])
        self.table_placeholder.dataframe(pd.DataFrame(self.epoch_data))


def main():
    st.title('Train and Test Your Model! :rocket:')
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ['Model Training', 'Image Prediction'])

    if selection == 'Model Training':
        # Get list of directories in 'datasets/food-101/images'
        image_dirs = [d.replace('_', ' ').title() for d in os.listdir('datasets/food-101/images') if os.path.isdir(os.path.join('datasets/food-101/images', d))]
        # Get user input for directories to train on
        dir1 = st.selectbox('Select the first directory', options=image_dirs)
        dir2 = st.selectbox('Select the second directory', options=image_dirs)
        if dir1 == dir2:
            st.warning('Please select two different directories.')
        else:
            model_training(dir1.replace(' ', '_').lower(), dir2.replace(' ', '_').lower())
    else:
        image_prediction()

def model_training(dir1, dir2):
    # Define ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Load images from folders
    train_generator = datagen.flow_from_directory(
        directory='datasets/food-101/images',
        classes=[dir1, dir2],
        target_size=(64, 64),
        batch_size=32,
        subset='training'
    )

    # Save class labels
    with open('class_labels.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

    validation_generator = datagen.flow_from_directory(
        directory='datasets/food-101/images',
        classes=[dir1, dir2],
        target_size=(64, 64),
        batch_size=32,
        subset='validation'
    )

    # Create a simple CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Get user input for number of epochs
    epochs = st.slider('Number of epochs', min_value=1, max_value=100, value=10)

    # Button to start training
    if st.button('Start Training'):
        # Train model and record history
        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            epochs=epochs,
            callbacks=[StreamlitCallback()]
        )

        # Plot accuracy progression
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Progression Over Epochs')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        st.pyplot(plt)

        # Delete the existing model file if it exists
        if os.path.exists('image_classifier.h5'):
            os.remove('image_classifier.h5')

        # Save the model for later use
        model.save('image_classifier.h5')
        st.success("Model trained and saved as image_classifier.h5")

def image_prediction():
    # Load class labels
    with open('class_labels.json', 'r') as f:
        class_labels = json.load(f)

    st.write(f"The current model has been trained against: {', '.join(class_labels.keys())}")

    model = load_model('image_classifier.h5')
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(64, 64))
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        class_idx = np.argmax(pred[0])
        # Get the class label from the class_labels dictionary
        class_label = list(class_labels.keys())[class_idx]
        st.success(f'The model has predicted this image as {class_label}.')

if __name__ == "__main__":
    main()
