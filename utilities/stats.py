import matplotlib.pyplot as plt
import utilities.configs as configs
import numpy as np
import utilities.model as model
import utilities.data as data

def a():
    print("a")

def dataset_count(ret = False):
    images_count, images_count_total = data.dataset_count()
    print('Total count: {}'.format(images_count_total))
    for key, value in images_count.items():
        print('{} count: {}'.format(key, value))
    plt.figure(figsize=(16,4))
    plt.bar(range(len(configs.labels)), list(images_count.values()))
    plt.xticks(range(len(configs.labels)), configs.labels)
    plt.title('Dataset distribution')
    plt.show()

    if(ret):
        return images_count

def dataset_probabilities(probabilities):
    plt.figure(figsize=(16,4))
    plt.bar(range(len(configs.labels)), list(probabilities[0]))
    plt.xticks(range(len(configs.labels)), configs.labels)
    plt.title('Probability distribution')
    plt.show()

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

def show_layer(model, n = 2):
    filters = model.layers[n].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = 6, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    plt.show()