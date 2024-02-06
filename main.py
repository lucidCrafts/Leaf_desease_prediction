import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

def main():
    # Define paths to the training and testing directories
    train_dir = "C:/DataScience/leaf_desease/leaf_dataset_extract/dataset/train"
    test_dir = 'C:/DataScience/leaf_desease/leaf_dataset_extract/dataset/test'

    # Define batch size and image size
    batch_size = 248
    target_size = (122, 122)

    # Use ImageDataGenerator for loading and preprocessing images
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        # Add padding to resize images to target size while preserving aspect ratio
        # Padding will be added as necessary to achieve the target size
        preprocessing_function=lambda img: tf.image.resize_with_pad(img, target_size[0], target_size[1])
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Load and preprocess training images
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'  # Use categorical mode for multi-class classification
    )

    # Load and preprocess testing images
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Set shuffle to False to preserve the order of filenames
    )

    # Load the pre-trained VGG16 model without the top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(target_size[0], target_size[1], 3))

    num_classes = 38

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Create a new sequential model
    model = Sequential()

    # Add the pre-trained VGG16 base model
    model.add(base_model)

    # Flatten the output of the base model
    model.add(Flatten())

    # Add custom dense layers for disease classification
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Adjust output to the number of disease classes

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',  # Use categorical crossentropy for multi-class classification
                metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Define a callback to save the model weights
    checkpoint_path = 'leaf_disease_detection_model.h5'
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                        monitor='val_accuracy',
                                        save_best_only=True,
                                        mode='max',
                                        verbose=1)

    # Train the model with data augmentation
    epochs = 1

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        callbacks=[checkpoint_callback]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
    print("Test Accuracy:", test_acc)

    # Load the saved model
    saved_model = tf.keras.models.load_model('leaf_disease_detection_model.h5')

    # Make predictions
    predictions = saved_model.predict(test_generator, steps=len(test_generator))

if __name__ == "__main__":
    main()
