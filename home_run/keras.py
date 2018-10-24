from .base import BaseServable
from keras.models import load_model


class KerasServable(BaseServable):
    """Servable based on a Keras Model"""

    def _build(self):
        # Load in the model from disk
        self.model = load_model(self.dlhub['files']['model'])

    def run(self, inputs, **parameters):
        return self.model.predict(inputs)
