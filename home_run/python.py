from home_run.base import BaseServable
import pickle as pkl
import importlib
import logging
logger = logging.getLogger(__name__)


class PythonStaticMethodServable(BaseServable):
    """Servable based on a Python static method"""

    def _build(self):
        # Get the settings
        my_module = self.servable['methods']['run']['method_details']['module']
        my_method = self.servable['methods']['run']['method_details']['method_name']

        # Load the function
        my_module_obj = importlib.import_module(my_module)
        self.function = getattr(my_module_obj, my_method)

        # Determine whether we need to unpack the inputs
        self.unpack = self.servable['methods']['run']['method_details'].get('unpack', False)

        # Get whether it is autobatched
        self.autobatch = self.servable['methods']['run']['method_details']['autobatch']
        logger.info('Made a static method {} from {} with{} autobatch'.format(
            my_method, my_module, '' if self.autobatch else 'out'
        ))

    def _run(self, inputs, **parameters):
        if self.autobatch:
            if self.unpack:
                return [self.function(*x, **parameters) for x in inputs]
            else:
                return [self.function(x, **parameters) for x in inputs]
        if self.unpack:
            return self.function(*inputs, **parameters)
        return self.function(inputs, **parameters)


class PythonClassMethodServable(BaseServable):

    def _build(self):
        # Get the settings
        with open(self.dlhub['files']['pickle'], 'rb') as fp:
            my_object = pkl.load(fp)
        logger.info('Loaded picked object: {}'.format(self.dlhub['files']['pickle']))

        # Determine whether we need to unpack the inputs
        self.unpack = self.servable['methods']['run']['method_details'].get('unpack', False)

        # Get the method to be run
        my_method = self.servable['methods']['run']['method_details']['method_name']
        self.function = getattr(my_object, my_method)
        logger.info('Made a static method {} from type: {}'.format(
            my_method, my_object.__class__.__name__
        ))

    def _run(self, inputs, **parameters):
        if self.unpack:
            return self.function(*inputs, **parameters)
        return self.function(inputs, **parameters)
