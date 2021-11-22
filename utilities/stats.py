import pathlib
import matplotlib.pyplot as plt
import configs
import numpy as np
import model

def dataset_count():
    dataset_dir_path = pathlib.Path(configs.dataset_dir)
    images_count_total = len(list(dataset_dir_path.glob('*/*.jpg')))
    print('Total count: {}'.format(images_count_total))
    images_count = {}
    for label in configs.labels:
        count = len(list(dataset_dir_path.glob('{}/*.jpg'.format(label))))
        images_count[label] = count
        print('{} count: {}'.format(label, count))
    plt.figure(figsize=(16,4))
    plt.bar(range(len(configs.labels)), list(images_count.values()))
    plt.xticks(range(len(configs.labels)), configs.labels)
    plt.title('Dataset distribution')
    plt.show()

    return images_count

def plot_dataset(dataset, batch_index = 1, count = 6):
    plt.figure(figsize=(10, 10))
    for images, onehot in dataset.take(batch_index):
        for i in range(count):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            index = np.argmax(np.array(onehot[i]), axis=0)
            plt.title(configs.labels[index])
            plt.axis("off")

def plot_augmented_dataset(dataset, batch_index = 1, count = 6):
    plt.figure(figsize=(10, 10))
    for images, _ in dataset.take(batch_index):
        for i in range(count):
            augim = model.augmentation_layers(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augim[i].numpy().astype("uint8"))
            plt.axis("off")