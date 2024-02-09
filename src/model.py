import tensorflow as tf

from funcy import identity, ljuxt, rcompose


def create_op():
    def Add():
        return tf.keras.layers.Add()

    def BatchNormalization():
        return tf.keras.layers.BatchNormalization()

    def Concatenate():
        return tf.keras.layers.Concatenate()

    def Conv(num_filters, kernel_size):
        return tf.keras.layers.Conv3D(num_filters, kernel_size, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())

    def MaxPooling():
        return tf.keras.layers.MaxPooling3D()

    def ReLU():
        return tf.keras.activations.relu

    def Sigmoid():
        return tf.keras.activations.sigmoid

    def UpConv():
        return tf.keras.layers.UpSampling3D()

    ###

    def ResidualUnit_0(num_filters):
        return rcompose(ljuxt(rcompose(Conv(num_filters // 4, 1),
                                       BatchNormalization(),
                                       ReLU(),
                                       Conv(num_filters // 4, 3),
                                       BatchNormalization(),
                                       ReLU(),
                                       Conv(num_filters, 1),
                                       BatchNormalization()),
                              rcompose(Conv(num_filters, 1),
                                       BatchNormalization())),
                        Add(),
                        ReLU())

    def ResidualUnit(num_filters):
        return rcompose(ljuxt(rcompose(Conv(num_filters // 4, 1),
                                       BatchNormalization(),
                                       ReLU(),
                                       Conv(num_filters // 4, 3),
                                       BatchNormalization(),
                                       ReLU(),
                                       Conv(num_filters, 1),
                                       BatchNormalization()),
                              identity),
                        Add(),
                        ReLU())

    def ResidualBlock(num_filters, num_units):
        def op(x):
            x = ResidualUnit_0(num_filters)(x)

            for _ in range(num_units - 1):
                x = ResidualUnit(num_filters)(x)

            return x

        return op

    def Encoder():
        def op(x):
            x = Conv(64, 3)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = ResidualBlock(64, 2)(x)
            y_4 = x

            x = MaxPooling()(x)
            x = ResidualBlock(128, 2)(x)
            y_3 = x

            x = MaxPooling()(x)
            x = ResidualBlock(256, 2)(x)
            y_2 = x

            x = MaxPooling()(x)
            x = ResidualBlock(512, 2)(x)
            y_1 = x

            x = MaxPooling()(x)
            x = ResidualBlock(1024, 2)(x)
            y_0 = x

            return y_0, y_1, y_2, y_3, y_4

        return op

    def Decoder():
        def op(x):
            x_0, x_1, x_2, x_3, x_4 = x

            x = x_0

            x = UpConv()(x)
            x = Concatenate()((x, x_1))
            x = ResidualBlock(512, 2)(x)

            x = UpConv()(x)
            x = Concatenate()((x, x_2))
            x = ResidualBlock(256, 2)(x)

            x = UpConv()(x)
            x = Concatenate()((x, x_3))
            x = ResidualBlock(128, 2)(x)

            x = UpConv()(x)
            x = Concatenate()((x, x_4))
            x = ResidualBlock(64, 2)(x)

            x = Conv(1, 1)(x)

            y = x

            return y

        return op

    return rcompose(Encoder(),
                    Decoder(),
                    Sigmoid())
