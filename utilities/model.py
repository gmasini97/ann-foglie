import tensorflow as tf
import configs
import stats

def weights():
    counts = stats.dataset_count()
    max_class = max(counts.values())
    class_weights = {}
    i = 0
    for k,v in counts.items():
        class_weights[i] = max_class/v
        i += 1

    return class_weights

def import_dataset(subset):
    return tf.keras.preprocessing.image_dataset_from_directory(
        configs.dataset_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=configs.labels,
        color_mode='rgb',
        batch_size=configs.BATCH_SIZE,
        image_size=configs.IMG_SIZE,
        shuffle=True,
        seed=configs.SEED,
        validation_split=configs.SPLIT,
        subset=configs.subset,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False
    )

def import_datasets():
    dataset_training = import_dataset('training')
    dataset_validation = import_dataset('validation')

    return (dataset_training, dataset_validation)

def augmentation_layers():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation(0.5, fill_mode='constant') # oppure "nearest"
        ]
    )

def size_add_channels(size):
    return size + (3,)

def compile(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

def fit(model, training, validation, weights, epochs=configs.EPOCHS):
    model.fit(
        training,
        epochs=epochs,
        validation_data=validation,
        class_weight=weights
    )