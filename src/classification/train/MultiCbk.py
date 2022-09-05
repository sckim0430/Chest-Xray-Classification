"""
Custom Save Checkpoint Class
"""
import keras


class MyCbk(keras.callbacks.Callback):
    """Custom Save CheckPoint Class

    Args:
        keras (class): keras.callbacks.Callback
    """

    def __init__(self, model, model_path):
        self.model_to_save = model
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save(self.model_path.format(epoch))
