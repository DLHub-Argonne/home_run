"""Tests related to the debug wrapper"""
from dlhub_sdk.models.servables.python import PythonStaticMethodModel
from pytest import fixture, raises

from home_run.python import PythonStaticMethodServable


def run(success: bool):
    print('Hello, pytest.')
    if success:
        return True
    else:
        raise ValueError('No!')


@fixture
def shim():
    model = PythonStaticMethodModel.from_function_pointer(run)
    model.set_name('test')
    model.set_title('Test')
    model.set_inputs('bool', 'Whether to run successfully')
    model.set_outputs('bool', 'Should always be True')
    return PythonStaticMethodServable(**model.to_dict())


def test_normal_execution(shim, capsys):
    result, metadata = shim.run(True)
    assert result
    assert metadata['success']
    assert metadata['stdout'] is None
    assert metadata['stderr'] is None
    assert 'wall_time' in metadata

    # Check to make sure some printing happened
    captured = capsys.readouterr()
    assert captured.out.startswith('Hello, pytest')


def test_debugging(shim):
    result, metadata = shim.run(True, debug=True)
    assert metadata['success']
    assert metadata['stdout'].startswith('Hello, pytest')
    assert metadata['stderr'] is None
    assert 'wall_time' in metadata


def test_error(shim):
    result, metadata = shim.run(False, debug=True)
    assert result is None
    assert not metadata['success']
    assert metadata['stdout'].startswith('Hello, pytest')
    assert 'No!' in metadata['error_message']

    with raises(ValueError) as exc:
        raise metadata['exc']
    assert str(exc.value) == 'No!'
