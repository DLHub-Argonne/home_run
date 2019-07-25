from dlhub_sdk.models.servables.pytorch import TorchModel
from home_run.pytorch import TorchServable
from unittest import TestCase
from torch import nn
import numpy as np
import torch
import os


class MultiNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 1)

    def forward(self, x, y):
        return self.layer(x), self.layer(y)


class TorchTest(TestCase):

    def setUp(self):
        # Make the single- and multi-input models
        self.model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
        )
        torch.save(self.model, 'model.pth')
        self.multimodel = MultiNetwork()
        torch.save(self.multimodel, 'multimodel.pth')

    def tearDown(self) -> None:
        os.unlink('model.pth')
        os.unlink('multimodel.pth')

    def test_single(self):
        model = TorchModel.create_model('model.pth', (None, 2), (None, 10)).set_name('test').set_title('test')

        # Make the servable
        servable = TorchServable(**model.to_dict())

        # Verify they yield the same results
        test_x = np.random.random((2, 2))
        y_true = self.model(torch.from_numpy(test_x).float()).detach().numpy()
        y_pred = servable.run(test_x)
        self.assertTrue(np.isclose(y_pred, y_true).all())

    def test_multimodel(self):
        model = TorchModel.create_model('multimodel.pth',
                                        [(None, 4)]*2, [(None, 10)]*2).set_name('test').set_title('test')

        # Make the servable
        servable = TorchServable(**model.to_dict())

        # Verify they yield the same results
        test_x = np.random.random((2, 4))
        test_y = np.random.random((4, 4))
        y_true = self.multimodel(torch.from_numpy(test_x).float(),
                                 torch.from_numpy(test_y).float())
        y_pred = servable.run((test_x, test_y))
        self.assertTrue(np.isclose(y_pred[0], y_true[0].detach()).all())
        self.assertTrue(np.isclose(y_pred[1], y_true[1].detach()).all())