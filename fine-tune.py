import tensorflow as tf 
import numpy as np
import os
import tensorflow_datasets as tfds 
import sys


BATCH_SIZE = 64
IMAGE_DIM = 224
IMAGE_CHANNELS = 3

def get_base_model(model_name):
    if model_name == 'mobilenetv2':
        pretrained_feature_extractor = tf.keras.applications.MobileNetV2(
            input_shape= (224,224,3),
            include_top = False,
            weights= 'imagenet',
            pooling='max'
        )
    elif model_name == 'vgg16':
        pretrained_feature_extractor = tf.keras.applications.VGG16(
            input_shape=(224,224,3),
            include_top= False,
            weights='imagenet',
            pooling='max'
        )
    elif model_name == 'resnet101':
        pretrained_feature_extractor = tf.keras.applications.ResNet101V2(
            input_shape=(224,224,3),
            include_top= False,
            weights='imagenet',
            pooling='max'
        )
    elif model_name == 'densenet121':
        pretrained_feature_extractor = tf.keras.applications.DenseNet121(
            input_shape=(224,224,3),
            include_top= False,
            weights='imagenet',
            pooling='max'
        )
    else:
        print("Base model not found")
        pretrained_feature_extractor = None
        
    return pretrained_feature_extractor


def get_model(base_model, classes):
    base_model.trainable = False
    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.Dense(classes)
        ]
    )
    
    return model

def format_image(img, label):
    image = tf.image.resize(img, (IMAGE_DIM, IMAGE_DIM))/255.0
    return image, label


def get_dataset():
    (train_examples, validation_examples), info = tfds.load(
        'cats_vs_dogs',
        split = ('train[:70%]', 'train[70%:]'),
        with_info=True,
        as_supervised=True
    )
    
    return train_examples, validation_examples, info

def process_dataset(train_examples, validation_examples, info):
    num_examples = info.splits['train'].num_examples
    train_batches = train_examples.cache().shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
    validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)
    
    return train_batches, validation_batches

def train(model, train_batches, validation_batches, num_epochs = 1):
    history = model.fit(train_batches, epochs = num_epochs, validation_data = validation_batches)
    print("Fine-tuning the model over cats vs dogs dataset complete.")
    return history

if __name__ == "__main__":
    feature_extractor = None
    if len(sys.argv) != 2:
        print("Wrong usage, Specify base model type.")
        exit(1)
    
    if sys.argv[1] == 'vgg16':
        print("Selected Base model type: VGG16")
        feature_extractor = get_base_model('vgg16')
        classifier = get_model(feature_extractor, 2)
    elif sys.argv[1] == 'mobilenet':
        print("Selected Base model type: MobilenetV2")
        feature_extractor = get_base_model('mobilenetv2')
        classifier = get_model(feature_extractor, 2)
    elif sys.argv[1] == 'resnet':
        print("Selected base model type: Resnet101")
        feature_extractor = get_base_model('resnet101')
        classifier = get_model(feature_extractor, 2)
    elif sys.argv[1] == 'densenet':
        print("Selected base model type: Densenet121")
        feature_extractor = get_base_model('densenet121')
        classifier = get_model(feature_extractor, 2)
    else:
        print("Selected base model not found in the script.")
        exit(2)
        
        
    classifier.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    train_set, validation_set, info = get_dataset()
    train_batches, validation_batches = process_dataset(train_set, validation_set, info)
    train(classifier, train_batches, validation_batches, num_epochs=2)
    
    
    classifier.save('./saved-model-unsigned')
    