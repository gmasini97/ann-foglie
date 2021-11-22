import tensorflow as tf
import utilities.configs as configs
import utilities.data as data

def a():
    print("a")

def weights():
    counts = data.dataset_count()
    max_class = max(counts.values())
    class_weights = {}
    i = 0
    for k,v in counts.items():
        class_weights[i] = max_class/v
        i += 1

    return class_weights

def augmentation_layers():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation(0.5, fill_mode='constant') # oppure "nearest"
        ]
    )

def early_stopping_callbacks():
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=configs.EARLY_STOPPING_PATIENCE)
    save_check_point = tf.keras.callbacks.ModelCheckpoint("./checkpoints/save_at_{epoch}")
    return (early_stopping, save_check_point)

def compile(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", "mse"],
    )

def fit(model, training, validation, weights, epochs=configs.EPOCHS):
    model.fit(
        training,
        epochs=epochs,
        validation_data=validation,
        class_weight=weights
    )