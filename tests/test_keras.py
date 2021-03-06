from dlhub_sdk.models.servables.keras import KerasModel
# Try importing straight keras first, as we assume if it's installed it is installed for a reason
try:
    import keras
except ImportError:
    from tensorflow import keras
from unittest import TestCase
from tempfile import mkdtemp
import numpy as np
import shutil
import os

from home_run.keras import KerasServable


def _make_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(16, input_shape=(1,), activation='relu', name='hidden'))
    model.add(keras.layers.Dense(1, name='output'))
    model.compile(optimizer='rmsprop', loss='mse')
    return model


class KerasTest(TestCase):

    def test_keras(self):
        # Make a Keras model
        model = _make_model()

        # Save it
        tempdir = mkdtemp()
        try:
            model_path = os.path.join(tempdir, 'model.hd5')
            model.save(model_path)

            # Make the model
            metadata = KerasModel.create_model(model_path, ["y"])
            metadata.set_title('Keras Test')
            metadata.set_name('mlp')

            # Make the servable
            servable = KerasServable(**metadata.to_dict())
            x = [[1]]
            self.assertAlmostEqual(model.predict(np.array(x))[0],
                                   servable.run(x)[0])

        finally:
            shutil.rmtree(tempdir)

    def test_keras_multiinput(self):
        # Make a Keras model
        input_1 = keras.layers.Input(shape=(1,))
        input_2 = keras.layers.Input(shape=(1,))
        inputs = keras.layers.Concatenate()([input_1, input_2])
        dense = keras.layers.Dense(16, activation='relu')(inputs)
        output = keras.layers.Dense(1, activation='linear')(dense)
        model = keras.models.Model(inputs=[input_1, input_2], outputs=output)
        model.compile(optimizer='rmsprop', loss='mse')

        # Save it
        tempdir = mkdtemp()
        try:
            model_path = os.path.join(tempdir, 'model.hd5')
            model.save(model_path)

            # Make the model
            metadata = KerasModel.create_model(model_path, ["y"])
            metadata.set_title('Keras Test')
            metadata.set_name('mlp')

            # Make the servable
            servable = KerasServable(**metadata.to_dict())
            x = [[1]]
            servable.run([x, x])
            self.assertAlmostEqual(model.predict([np.array(x)]*2)[0],
                                   servable.run([x, x])[0])
        finally:
            shutil.rmtree(tempdir)

    def test_keras_multioutput(self):
        # Make a Keras model
        input_1 = keras.layers.Input(shape=(1,))
        dense = keras.layers.Dense(16, activation='relu')(input_1)
        output_1 = keras.layers.Dense(1, activation='linear')(dense)
        output_2 = keras.layers.Dense(1, activation='linear')(dense)
        model = keras.models.Model(inputs=input_1, outputs=[output_1, output_2])
        model.compile(optimizer='rmsprop', loss='mse')

        # Save it
        tempdir = mkdtemp()
        try:
            model_path = os.path.join(tempdir, 'model.hd5')
            model.save(model_path)

            # Make the model
            metadata = KerasModel.create_model(model_path, ["y"])
            metadata.set_title('Keras Test')
            metadata.set_name('mlp')

            # Make the servable
            servable = KerasServable(**metadata.to_dict())
            x = [[1]]
            servable.run(x)
            self.assertTrue(np.isclose(model.predict(np.array(x)), servable.run(x)).all())
        finally:
            shutil.rmtree(tempdir)

    def test_keras_multifile(self):
        """Test loading a shim when model is saved as a separate arch and weights file"""

        # Make a Keras model
        model = _make_model()

        # Save it
        tempdir = mkdtemp()
        try:
            model_path = os.path.join(tempdir, 'model.hd5')
            model.save(model_path, include_optimizer=False)
            model_json = os.path.join(tempdir, 'model.json')
            with open(model_json, 'w') as fp:
                print(model.to_json(), file=fp)
            model_yaml = os.path.join(tempdir, 'model.yml')
            with open(model_yaml, 'w') as fp:
                print(model.to_yaml(), file=fp)

            weights_path = os.path.join(tempdir, 'weights.hd5')
            model.save_weights(weights_path)

            # Create the metadata
            metadata = KerasModel.create_model(weights_path, ['y'], arch_path=model_path)
            metadata.set_title('Keras Test')
            metadata.set_name('mlp')

            # Make the servable
            servable = KerasServable(**metadata.to_dict())
            x = [[1]]
            self.assertAlmostEqual(model.predict(np.array(x))[0],
                                   servable.run(x)[0])

            # Test with the JSON and YAML
            metadata = KerasModel.create_model(weights_path, ['y'], arch_path=model_json)
            metadata.set_title('Keras Test').set_name('mlp')
            servable = KerasServable(**metadata.to_dict())
            self.assertAlmostEqual(model.predict(np.array(x))[0], servable.run(x)[0])

            metadata = KerasModel.create_model(weights_path, ['y'], arch_path=model_yaml)
            metadata.set_title('Keras Test').set_name('mlp')
            servable = KerasServable(**metadata.to_dict())
            self.assertAlmostEqual(model.predict(np.array(x))[0], servable.run(x)[0])
        finally:
            shutil.rmtree(tempdir)

    def test_custom_layers(self):
        """Test adding custom layers to the definition"""

        # Make a simple model
        model = _make_model()

        tmpdir = mkdtemp()
        try:
            # Save it
            model_path = os.path.join(tmpdir, 'model.hd5')
            model.save(model_path)

            # Create the metadata
            metadata = KerasModel.create_model(model_path, ['y'],
                                               custom_objects={'Dense': keras.layers.Dense})
            metadata.set_title('Keras Test')
            metadata.set_name('mlp')

            # Make the servable
            servable = KerasServable(**metadata.to_dict())
            x = [[1]]
            self.assertAlmostEqual(model.predict(np.array(x))[0],
                                   servable.run(x)[0])
        finally:
            shutil.rmtree(tmpdir)
