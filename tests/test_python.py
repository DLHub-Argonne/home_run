from dlhub_sdk.models.servables.python import PythonStaticMethodModel, PythonClassMethodModel
from dlhub_sdk.utils.schemas import validate_against_dlhub_schema
from dlhub_sdk.utils.types import compose_argument_block
from home_run.version import __version__
from unittest import TestCase
from tempfile import mkstemp
from platform import system
import pickle as pkl
import numpy as np
import os

from home_run.python import PythonStaticMethodServable, PythonClassMethodServable


def multifile_input(input, inputs, const):
    return all([os.path.isfile(input),
                all(map(os.path.isfile, inputs)),
                const])


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
        self.assertTrue(np.isclose([2, 4], servable.run([[1, 2], [3, 4]],
                                                        parameters=dict(axis=1))).all())

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
            self.assertAlmostEqual(4, servable.run(1, parameters=dict(b=2)))

            # Test operations for the base class
            self.assertEqual(__version__, servable.get_version())
            self.assertEqual(model.to_dict(), servable.get_recipe())

        finally:
            os.unlink(filename)

    def test_multiargs(self):
        # Make the maximum function
        model = PythonStaticMethodModel.from_function_pointer(max)\
            .set_name('test').set_title('test')

        # Describe the inputs
        model.set_inputs('tuple', 'Two numbers',
                         element_types=[
                             compose_argument_block('float', 'A number'),
                             compose_argument_block('float', 'A second number')
                         ])
        model.set_outputs('float', 'Maximum of the two numbers')
        model.set_unpack_inputs(True)

        # Make sure the shim works
        servable = PythonStaticMethodServable(**model.to_dict())
        self.assertEquals(servable.run((1, 2)), 2)

    def test_multiargs_autobatch(self):
        # Make the maximum function
        model = PythonStaticMethodModel.from_function_pointer(max, autobatch=True)\
            .set_name('test').set_title('test')

        # Describe the inputs
        model.set_inputs('list', 'List of pairs of numbers',
                         item_type=compose_argument_block(
                             'tuple', 'Two numbers',
                             element_types=[
                                 compose_argument_block('float', 'A number'),
                                 compose_argument_block('float', 'A second number')
                             ]))
        model.set_outputs('list', 'Maximum of each pair', item_type='float')
        model.set_unpack_inputs(True)

        # Make sure the shim works
        servable = PythonStaticMethodServable(**model.to_dict())
        self.assertEquals(servable.run([(1, 2)]), [2])

    def test_multiargs_pickle(self):
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
            model.set_inputs('tuple', 'inputs',
                             element_types=[compose_argument_block('float', 'Number')]*2)
            model.set_outputs('float', 'Output')
            model.set_unpack_inputs(True)

            # Make the servable
            validate_against_dlhub_schema(model.to_dict(), 'servable')
            servable = PythonClassMethodServable(**model.to_dict())

            # Test the servable
            self.assertAlmostEqual(4, servable.run([1, 2]))

        finally:
            os.unlink(filename)

    def test_single_file_input(self):
        # Make the metadata model
        model = PythonStaticMethodModel.from_function_pointer(os.path.isfile).set_name('test')
        model.set_title('test')
        model.set_inputs('file', 'A file')
        model.set_outputs('boolean', 'Whether it exists')

        # Make the servable
        servable = PythonStaticMethodServable(**model.to_dict())

        # Run on local file
        self.assertTrue(servable.run({'url': __file__}))
        if system() != 'Windows':
            self.assertTrue(servable.run({'url': 'file:///' + __file__}))  # Fail on Windows

        # Run on remote file
        self.assertTrue(servable.run({'url': 'https://www.google.com/images/branding/'
                                             'googlelogo/1x/googlelogo_color_272x92dp.png'}))

    def test_single_file_list_input(self):
        # Make the metadata model
        model = PythonStaticMethodModel.from_function_pointer(os.path.isfile, autobatch=True)
        model.set_name('test')
        model.set_title('test')
        model.set_inputs('list', 'List of files', item_type='file')
        model.set_outputs('list', 'Whether each file exists', item_type='boolean')

        # Make the servable
        servable = PythonStaticMethodServable(**model.to_dict())

        # Run on local file
        self.assertTrue(servable.run([{'url': __file__}]))
        if system() != 'Windows':
            self.assertTrue(servable.run([{'url': 'file:///' + __file__}]))  # Fail on Windows

        # Run on remote file
        self.assertTrue(servable.run([{'url': 'https://www.google.com/images/branding/'
                                              'googlelogo/1x/googlelogo_color_272x92dp.png'}]))

    def test_file_multiinput(self):
        model = PythonStaticMethodModel.from_function_pointer(multifile_input)
        model.set_name('test')
        model.set_title('test')
        model.set_inputs('tuple', 'Several things', element_types=[
            compose_argument_block('file', 'Single file'),
            compose_argument_block('list', 'Multiple files', item_type='file'),
            compose_argument_block('boolean', 'Something random')
        ])
        model.set_outputs('bool', 'Should be True')
        model.set_unpack_inputs(True)

        # Make the servable
        servable = PythonStaticMethodServable(**model.to_dict())

        # Test it
        self.assertTrue(servable.run([
            {'url': __file__},
            [{'url': __file__}],
            True
        ]))
