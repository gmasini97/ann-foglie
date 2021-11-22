import os
import tensorflow as tf

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/model.h5'))

    def predict(self, X):
        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)
        return out