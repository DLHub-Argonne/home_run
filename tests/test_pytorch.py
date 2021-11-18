from dlhub_sdk.models.servables.pytorch import TorchModel
from pytest import fixture
from torch import nn
import numpy as np
import torch

from home_run.pytorch import TorchServable


class MultiNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 1)

    def forward(self, x, y):
        return self.layer(x), self.layer(y)


@fixture()
def model(tmpdir):
    # Make the single- and multi-input models
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
    )
    model_path = str(tmpdir / 'model.pth')
    torch.save(model, model_path)
    return model, model_path


@fixture()
def multimodel(tmpdir):
    multimodel = MultiNetwork()
    model_path = str(tmpdir / 'multimodel.pth')
    torch.save(multimodel, model_path)
    return multimodel, model_path


def test_single(model):
    # Create the metadata model
    model, path = model
    metadata = TorchModel.create_model(path, (None, 2), (None, 10)) \
        .set_name('test').set_title('test')

    # Make the servable
    servable = TorchServable(**metadata.to_dict())

    # Verify they yield the same results
    test_x = np.random.random((2, 2))
    y_true = model(torch.from_numpy(test_x).float()).detach().numpy()
    y_pred, _ = servable.run(test_x)
    assert np.isclose(y_pred, y_true).all()


def test_multimodel(multimodel):
    model, path = multimodel
    metadata = TorchModel.create_model(path, [(None, 4)] * 2, [(None, 10)] * 2) \
        .set_name('test').set_title('test')

    # Make the servable
    servable = TorchServable(**metadata.to_dict())

    # Verify they yield the same results
    test_x = np.random.random((2, 4))
    test_y = np.random.random((4, 4))
    y_true = model(torch.from_numpy(test_x).float(),
                   torch.from_numpy(test_y).float())
    y_pred, _ = servable.run((test_x, test_y))
    assert np.isclose(y_pred[0], y_true[0].detach()).all()
    assert np.isclose(y_pred[1], y_true[1].detach()).all()
