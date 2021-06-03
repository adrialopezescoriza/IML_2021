import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

height = 96
width = 96


def load_image(img, training):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.keras.applications.mobilenet.preprocess_input(tf.cast(img, tf.float32))
    img = tf.image.resize(img, (height, width))
    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
    return img


def load_triplets(triplet, training):
    ids = tf.strings.split(triplet)
    anchor = load_image(tf.io.read_file('food/' + ids[0] + '.jpg'), training)
    truthy = load_image(tf.io.read_file('food/' + ids[1] + '.jpg'), training)
    falsy = load_image(tf.io.read_file('food/' + ids[2] + '.jpg'), training)
    if training:
        return tf.stack([anchor, truthy, falsy], axis=0), 1
    else:
        return tf.stack([anchor, truthy, falsy], axis=0)


def split_train_triplets():
    samples = 'train_triplets.txt'
    with open(samples, 'r') as file:
        triplets = [line for line in file.readlines()]
    train_samples, val_samples = train_test_split(triplets, test_size=0.2)
    with open('val_samples.txt', 'w') as file:
        for item in val_samples:
            file.write(item)
    with open('train_samples.txt', 'w') as file:
        for item in train_samples:
            file.write(item)
    return len(train_samples)


def make_dataset(dataset_filename, training=True):
    dataset = tf.data.TextLineDataset(dataset_filename)
    dataset = dataset.map(lambda triplet: load_triplets(triplet, training))
    return dataset


def create_model():
    inputs = tf.keras.Input(shape=(3, height, width, 3))
    pretrained = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(height, width, 3))
    pretrained.trainable = False # Train only top layers
    transfer = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))
    ])
    anchor, truthy, falsy = inputs[:, 0, ...], inputs[:, 1, ...], inputs[:, 2, ...]
    anchor_features = transfer(pretrained(anchor))
    truthy_features = transfer(pretrained(truthy))
    falsy_features = transfer(pretrained(falsy))
    embeddings = tf.stack([anchor_features, truthy_features, falsy_features], axis=-1)
    full_model = tf.keras.Model(inputs=inputs, outputs=embeddings)
    #full_model.summary()
    return full_model


def create_prediction_model(model):
    distance_truthy, distance_falsy = distance(model.output)
    predictions = tf.cast(tf.greater_equal(distance_falsy, distance_truthy), tf.int8) # =1 if dist(false,anchor) >= dist(truth,anchor)
    return tf.keras.Model(inputs=model.inputs, outputs=predictions)


def distance(embeddings):
    anchor, truthy, falsy = embeddings[..., 0], embeddings[..., 1], embeddings[..., 2]
    distance_truthy = tf.reduce_sum(tf.square(anchor - truthy), 1)
    distance_falsy = tf.reduce_sum(tf.square(anchor - falsy), 1)
    return distance_truthy, distance_falsy


# adapt trainable part of NN so that the similarity between the learned embeddings of
# anchor and truth is greater than anchor and false
def loss_func(_, embeddings): # based on only y_pred and not y_true
    distance_truthy, distance_falsy = distance(embeddings)
    return tf.reduce_mean(tf.math.softplus(distance_truthy - distance_falsy))


def prediction_accuracy(_, embeddings): # based on only y_pred and not y_true
    distance_truthy, distance_falsy = distance(embeddings)
    return tf.reduce_mean(tf.cast(tf.greater_equal(distance_falsy, distance_truthy), tf.float32))


num_epochs = 5
train_batch_size = 32
predict_batch_size = 256

num_train_samples = split_train_triplets()
train_dataset = make_dataset('train_samples.txt')
val_dataset = make_dataset('val_samples.txt')
model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=loss_func,
                metrics=[prediction_accuracy])
train_dataset = train_dataset.shuffle(1024, reshuffle_each_iteration=True).repeat().batch(train_batch_size)
val_dataset = val_dataset.batch(train_batch_size)
history = model.fit(
    train_dataset,
    steps_per_epoch=int(np.ceil(num_train_samples / train_batch_size)),
    epochs=num_epochs,
    validation_data=val_dataset,
    validation_steps=10
)
test_dataset = make_dataset('test_triplets.txt', training=False).batch(predict_batch_size).prefetch(2)
prediction_model = create_prediction_model(model)
num_test_samples = 59544
predictions = prediction_model.predict(
    test_dataset,
    steps=int(np.ceil(num_test_samples / predict_batch_size)),
    verbose=1)
np.savetxt('predictions.txt', predictions, fmt='%i')

