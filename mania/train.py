import argparse
import random

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from scipy.misc import imread, imresize, imsave

from simplemodel import simple_model


class DataSequence(Sequence):
    def __init__(self,
                 image_filenames: list,
                 image_shape: tuple = (84, 200, 3),
                 batch_size: int = 32,
                 classes: int = 1,
                 shuffle: bool = True):

        self.batch_size = batch_size
        self.image_filenames = image_filenames
        self.classes = classes

        self.image_shape = image_shape

        self.shuffle = shuffle
        self.batches_fetched = 0
        self.debug = False

        self.len_images = len(self.image_filenames)

    def __len__(self):
        return np.math.ceil(self.len_images / self.batch_size)

    def __getitem__(self, index):
        s = index * self.batch_size
        e = s + self.batch_size
        self.batches_fetched += 1

        if not (self.batches_fetched % len(self)) and self.shuffle:
            random.shuffle(self.image_filenames)

        filenames_batch = [self.image_filenames[i % self.len_images] for i in range(s, e)]

        x_batch = np.zeros((self.batch_size,) + self.image_shape)
        y_batch = np.zeros((self.batch_size))

        for i, image_fn in enumerate(filenames_batch):
            img = imread(image_fn, mode='RGB')

            # # horizontal flip
            # if random.uniform(0, 1) > 0.5:
            #     np.flip(img, 1)
            #
            # # random crop
            # h, w = img.shape[0:2]
            # x1 = random.randint(0, w // 10)
            # x2 = w - random.randint(0, w // 10)
            # y1 = random.randint(0, h // 10)
            # y2 = h - random.randint(0, h // 10)
            # img = img[y1:y2, x1:x2]
            #
            # # rotate
            # angle = random.uniform(-10, 10)
            # img = imrotate(img, angle)

            # final resize
            img = imresize(img, self.image_shape)
            if self.debug:
                imsave('/tmp/%s.jpg' % i, img)

            x_batch[i] = img

            s = image_fn.split('/')[-1].find("positive")
            y_batch[i] = 1 if s >= 0 else 0

        x_batch = x_batch / 127.5 - 1

        return x_batch, y_batch


def main(train_file, val_file):
    with open(train_file) as f:
        train_dataset = [s.strip() for s in f]
    with open(val_file) as f:
        val_dataset = [s.strip() for s in f]

    print('Train examples %s, validation examples %s' % (len(train_dataset), len(val_dataset)))

    classes = 1

    train_generator = DataSequence(train_dataset, classes=classes)
    val_generator = DataSequence(val_dataset, classes=classes)

    weights_filename = 'checkpoints/uaplates.h5'
    model = simple_model()
    model.summary()

    try:
        model.load_weights(weights_filename)
    except:
        pass

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=1000,
        epochs=1000,
        callbacks=[ModelCheckpoint(weights_filename, save_best_only=False)],
        validation_data=val_generator,
        validation_steps=100,
        workers=4,
        use_multiprocessing=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('-train_dataset', default='dataset/train.txt', help='Train dataset list  directory')
    parser.add_argument('-val_dataset', default='dataset/val.txt', help='Validation percentage')
    args = parser.parse_args()

    main(args.train_dataset, args.val_dataset)
