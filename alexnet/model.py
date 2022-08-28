from tensorflow import keras
from tensorflow.keras import layers

class BaseModel(object):

    def __init__(self, num_channels: int = 3, w_size: int = 224, h_size: int = 224, num_classes: int = 10):
        self.num_channels = num_channels
        self.w_size = w_size
        self.h_size = h_size
        self.num_classes = num_classes
        
    # def _define_model(self):
        
    #     model = keras.Sequential(
    #         [   layers.Input((self.w_size, self.h_size, self.num_channels)),
    #             layers.Conv2D(filters=6, kernel_size=(5,5), padding='valid', activation='relu', input_shape=(224, 224, 3)),
    #             layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same'),
    #             layers.Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='relu'),
    #             layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same'),
    #             layers.Flatten(),
    #             layers.Dense(120, activation='relu'),
    #             layers.Dense(84, activation='relu'),
    #             layers.Dense(10, activation='softmax'),
    #         ])

    #     return model
    def _define_model(self):
    
        model = keras.Sequential(
            [   layers.Input((self.w_size, self.h_size, self.num_channels)),
                #layer1
                layers.Conv2D(filters=96, kernel_size=(11,11),strides=4, padding='valid', activation='relu', input_shape=(224, 224, 3)),
                layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same'),
                layers.Normalization(),
                #layer2
                layers.Conv2D(filters=256, kernel_size=(3,3),strides=1, padding='same', activation='relu'),
                layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same'),
                layers.Normalization(),
                #layer3
                layers.Conv2D(filters=384, kernel_size=(3,3),strides=1, padding='same', activation='relu'),
                #layer4
                layers.Conv2D(filters=384, kernel_size=(3,3),strides=1, padding='same', activation='relu'),
                #layer5
                layers.Conv2D(filters=256, kernel_size=(3,3),strides=1, padding='same', activation='relu'),
                layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same'),        
                layers.Flatten(),
                #fc layer1
                layers.Dense(4096, activation='relu'),
                layers.Dropout(0.5),
                #fc layer2
                layers.Dense(4096, activation='relu'),
                layers.Dropout(0.5),
                #output layer
                layers.Dense(10, activation='softmax'),
            ])

        return model
    
    def __call__(self, loss, metrics):
        model = self._define_model()

        model.compile(
            optimizer = keras.optimizers.Adam(),
            loss = loss,
            metrics = metrics,
        )

        return model
