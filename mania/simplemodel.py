from keras import Model
from keras.layers import Dense, Input, Dropout, Conv2D, LeakyReLU, Flatten
from keras.utils import plot_model


def simple_model(image_shape: tuple = (84, 200, 3)):
    """
    Create model on top of pretrained feature extractor
    :param image_shape: image size tuple, (h, w, d)
    :param train_head_only: freeze feature extractor layers if True, use Adam; train all network otherwise with SGD
    :return: keras model instance
    """
    input_image = Input(image_shape)
    x = Conv2D(16, 3)(input_image)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(32, 3)(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)

    x = Dense(20, activation='relu')(x)
    x = Dense(20, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    m = Model(inputs=input_image, outputs=x)

    optimizer = 'adam'
    m.compile(optimizer, 'binary_crossentropy', metrics=['accuracy'])

    return m

if __name__ == '__main__':
    model = simple_model()

    model.summary()

    plot_model(model, show_shapes=True)
