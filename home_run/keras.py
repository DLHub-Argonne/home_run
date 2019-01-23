from .base import BaseServable
from keras.models import load_model
import numpy as np


class KerasServable(BaseServable):
    """Servable based on a Keras Model"""

    def _build(self):
        # Load in the model from disk
        self.model = load_model(self.dlhub['files']['model'])

        # Check whether this model is a multi-input
        self.is_multiinput = self.servable['methods']['run']['input']['type'] == 'tuple'
        self.is_multioutput = self.servable['methods']['run']['output']['type'] == 'tuple'

    def _run(self, inputs, **parameters):
        if self.is_multiinput:
            # If multiinput, provide a list of numpy arrays
            X = [np.array(x) for x in inputs]
        else:
            # If not, just turn the inputs into an array
            X = np.array(inputs)

        # Run the model
        result = self.model.predict(X)

        # Convert results to list so they can be used by non-Python clients
        if self.is_multioutput:
            return [y.tolist() for y in result]
        else:
            return result.tolist()
