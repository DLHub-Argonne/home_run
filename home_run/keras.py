from .base import BaseServable
from keras.models import load_model
import numpy as np


class KerasServable(BaseServable):
    """Servable based on a Keras Model"""

    def _build(self):
        # Load in the model from disk
        self.model = load_model(self.dlhub['files']['model'])

        # Check whether this model is a multi-input
        self.is_multiinput = self.servable['methods']['run']['input']['type'] == 'list'

    def _run(self, inputs, **parameters):
        if self.is_multiinput:
            # If multiinput, provide a list of numpy arrays
            X = [np.array(x) for x in inputs]
        else:
            # If not, just turn the inputs into an array
            X = np.array(inputs)

        return self.model.predict(X).tolist()
