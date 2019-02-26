from .base import BaseServable
from keras.models import load_model, model_from_yaml, model_from_json
from importlib import import_module
import numpy as np
import logging
logger = logging.getLogger(__name__)


class KerasServable(BaseServable):
    """Servable based on a Keras Model"""

    def _build(self):
        # Get all of the custom objects
        if 'options' in self.servable and 'custom_objects' in self.servable['options']:
            custom_objects = dict(self.servable['options']['custom_objects'])

            # Import each of the described classes
            for k, v in custom_objects.items():
                temp = v.split(".")
                mod = import_module('.'.join(temp[:-1]))
                cls = getattr(mod, temp[-1])
                custom_objects[k] = cls
            logger.info('Defined custom layers: {}'.format(', '.join(custom_objects.keys())))
        else:
            custom_objects = None

        # Load in the model from disk
        if 'arch' in self.dlhub['files']:
            # Load in the architecture
            arch_path = self.dlhub['files']['arch']
            if arch_path.endswith('.h5') or arch_path.endswith('.hdf') \
                    or arch_path.endswith('.hdf5') or arch_path.endswith('.hd5'):
                self.model = load_model(arch_path, custom_objects=custom_objects, compile=False)
            elif arch_path.endswith('.json'):
                with open(arch_path) as fp:
                    json_string = fp.read()
                self.model = model_from_json(json_string, custom_objects=custom_objects)
            elif arch_path.endswith('.yml') or arch_path.endswith('.yaml'):
                with open(arch_path) as fp:
                    yaml_string = fp.read()
                self.model = model_from_yaml(yaml_string, custom_objects=custom_objects)
            else:
                raise ValueError('File type for architecture not recognized')

            # Load in the weights
            self.model.load_weights(self.dlhub['files']['model'])
            logger.info('Loaded arch ({}) and weights ({})'.format(arch_path,
                                                                   self.dlhub['files']['model']))
        else:
            self.model = load_model(self.dlhub['files']['model'])
            logger.info('Loaded single-file model ({})'.format(self.dlhub['files']['model']))

        # Check whether this model is a multi-input
        self.is_multiinput = self.servable['methods']['run']['input']['type'] == 'tuple'
        self.is_multioutput = self.servable['methods']['run']['output']['type'] == 'tuple'
        logger.info('Detected multiinput={} and multioutput={}'.format(
            self.is_multiinput, self.is_multioutput
        ))

    def _run(self, inputs, **parameters):
        if self.is_multiinput:
            # If multiinput, provide a list of numpy arrays
            X = [np.array(x) for x in inputs]
            logger.debug('Received {} arrays with shapes: {}'.format(
                len(X), [i.shape for i in X]
            ))
        else:
            # If not, just turn the inputs into an array
            X = np.array(inputs)
            logger.debug('Received array with shape: {}'.format(X.shape))

        # Run the model
        result = self.model.predict(X)

        # Convert results to list so they can be used by non-Python clients
        if self.is_multioutput:
            return [y.tolist() for y in result]
        else:
            return result.tolist()
