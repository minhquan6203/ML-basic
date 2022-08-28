from tensorflow import keras
from tensorflow.keras import layers

class BaseModel(object):

    def __init__(self, num_channels: int = 3, w_size: int = 224, h_size: int = 224, num_classes: int = 10):
        self.num_channels = num_channels
        self.w_size = w_size
        self.h_size = h_size
        self.num_classes = num_classes
        
    def _define_model(self):
        
        model = keras.Sequential(
            [   layers.Input((self.w_size, self.h_size, self.num_channels)),
                #layer 1
                layers.Conv2D(64, (3, 3), activation='relu', padding='same',input_shape=(224, 224, 3)),
                layers.Conv2D( 64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2), strides=(2, 2)),

                # layer 2
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2), strides=(2, 2)),

                # layer 3
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2), strides=(2, 2)),

                # layer 4
                layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2), strides=(2, 2)),

                # layer 5
                layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2), strides=(2, 2)),

                layers.Flatten(),
                # fc layer 1
                layers.Dense(4096, activation='relu'),
                #fc layer 2
                layers.Dense(4096, activation='relu'),
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
