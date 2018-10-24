from home_run.base import BaseServable
import pickle as pkl
import importlib


class PythonStaticMethodServable(BaseServable):
    """Servable based on a Python static method"""

    def _build(self):
        # Get the settings
        my_module = self.servable['methods']['run']['method_details']['module']
        my_method = self.servable['methods']['run']['method_details']['method_name']

        # Load the function
        my_module = importlib.import_module(my_module)
        self.function = getattr(my_module, my_method)

        # Get whether it is autobatched
        self.autobatch = self.servable['methods']['run']['method_details']['autobatch']

    def _run(self, inputs, **parameters):
        if self.autobatch:
            return [self.function(x, **parameters) for x in inputs]
        return self.function(inputs, **parameters)


class PythonClassMethodServable(BaseServable):

    def _build(self):
        # Get the settings
        with open(self.dlhub['files']['pickle'], 'rb') as fp:
            my_object = pkl.load(fp)

        # Get the method to be run
        my_method = self.servable['methods']['run']['method_details']['method_name']
        self.function = getattr(my_object, my_method)

    def _run(self, inputs, **parameters):
        return self.function(inputs, **parameters)
