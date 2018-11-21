from dlhub_sdk.models.servables.python import PythonStaticMethodModel
from home_run.python import PythonStaticMethodServable
from home_run import create_servable
from unittest import TestCase


class TestLoader(TestCase):

    def test_loader(self):
        # Make an example static method
        model = PythonStaticMethodModel.create_model('numpy', 'max', autobatch=False,
                                                     function_kwargs={'axis': 0})
        model.set_title('Example function')
        model.set_name('function')
        model.set_inputs('ndarray', 'Matrix', shape=[None, None])
        model.set_outputs('ndarray', 'Max of a certain axis', shape=[None])

        # Test the loader
        servable = create_servable(model.to_dict())
        self.assertIsInstance(servable, PythonStaticMethodServable)
