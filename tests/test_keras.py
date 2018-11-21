from dlhub_sdk.models.servables.keras import KerasModel
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate
from unittest import TestCase
from tempfile import mkdtemp
import numpy as np
import shutil
import os

from home_run.keras import KerasServable


class KerasTest(TestCase):

    def test_keras(self):
        # Make a Keras model
        model = Sequential()
        model.add(Dense(16, input_shape=(1,), activation='relu', name='hidden'))
        model.add(Dense(1, name='output'))
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
            self.assertAlmostEqual(model.predict(np.array(x))[0],
                                   servable.run(x)[0])

        finally:
            shutil.rmtree(tempdir)

    def test_keras_multiinput(self):
        # Make a Keras model
        input_1 = Input(shape=(1,))
        input_2 = Input(shape=(1,))
        inputs = Concatenate()([input_1, input_2])
        dense = Dense(16, activation='relu')(inputs)
        output = Dense(1, activation='linear')(dense)
        model = Model(inputs=[input_1, input_2], outputs=output)
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
