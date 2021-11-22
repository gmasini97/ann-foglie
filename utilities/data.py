import utilities.configs as configs
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def get_first_of_type(dataset, label):
    plt.figure(figsize=(10, 10))
    for images, onehot in dataset.take(1):
        for i, image in enumerate(images):
            index = np.argmax(np.array(onehot[i]), axis=0)
            l = configs.labels[index]
            if(l != label):
                continue
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(l)
            plt.axis("off")
            return image

def import_dataset(subset = 'training'):
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
        subset=subset,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False
    )

def import_datasets():
    dataset_training = import_dataset('training')
    dataset_validation = import_dataset('validation')

    return (dataset_training, dataset_validation)