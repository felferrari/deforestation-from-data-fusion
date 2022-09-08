import tensorflow as tf

class ResetGenerator(tf.keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        print(epoch)
     

        