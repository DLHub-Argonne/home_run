from dlhub_sdk.models.servables.python import PythonStaticMethodModel, PythonClassMethodModel
from home_run import __version__
from unittest import TestCase
from tempfile import mkstemp
import pickle as pkl
import numpy as np
import os

from home_run.python import PythonStaticMethodServable, PythonClassMethodServable


class ExampleClass:

    def __init__(self, a):
        self.a = a

    def f(self, x, b=1):
        return self.a * x + b


class TestPython(TestCase):

    def test_static(self):
        # Make an example static method
        model = PythonStaticMethodModel.create_model('numpy', 'max', autobatch=False,
                                                     function_kwargs={'axis': 0})
        model.set_title('Example function')
        model.set_name('function')
        model.set_inputs('ndarray', 'Matrix', shape=[None, None])
        model.set_outputs('ndarray', 'Max of a certain axis', shape=[None])

        # Make the servable
        servable = PythonStaticMethodServable(**model.to_dict())

        # Test it
        self.assertTrue(np.isclose([3, 4], servable.run([[1, 2], [3, 4]])).all())

        # Test giving it parameters
        self.assertTrue(np.isclose([2, 4], servable.run([[1, 2], [3, 4]], axis=1)).all())

        # Test the autobatch
        model['servable']['methods']['run']['method_details']['autobatch'] = True
        servable = PythonStaticMethodServable(**model.to_dict())

        self.assertTrue(np.isclose([2, 4], servable.run([[1, 2], [3, 4]])).all())

    def test_pickle(self):
        # Make an example class
        x = ExampleClass(2)

        # Save a pickle
        fp, filename = mkstemp('.pkl')
        os.close(fp)
        try:
            with open(filename, 'wb') as fp:
                pkl.dump(x, fp)

            # Make the metadata file
            model = PythonClassMethodModel.create_model(filename, 'f')
            model.set_title('Example function')
            model.set_name('function')
            model.set_inputs('float', 'Input')
            model.set_outputs('float', 'Output')

            # Make the servable
            servable = PythonClassMethodServable(**model.to_dict())

            # Test the servable
            self.assertAlmostEqual(3, servable.run(1))
            self.assertAlmostEqual(4, servable.run(1, b=2))

            # Test operations for the base class
            self.assertEqual(__version__, servable.get_version())
            self.assertEqual(model.to_dict(), servable.get_recipe())

        finally:
            os.unlink(filename)
