import importlib

__version__ = '0.0.1'


def create_servable(recipe):
    """Given a recipe, create a servable

    Args:
        recipe (dict): Recipe describing a servable to be created
    Returns:
        (BaseServable): Servable object
    """

    # Technical note: We require the full path (e.g., "python.PythonStaticMethodServable")
    #  rather than just the name of the class because each submodule may contain imports that we
    #  do not want to install in a particular container. A routine that matches a class name
    #  to a full path will likely require importing modules, which will error if the
    #  imports in the module are not installed. So, listing the full path lets us avoid
    #  needing to install those modules to prevent import failures - leading to lighter containers.
    #  Using try/catch to only load modules that load properly will mask real import errors.
    #  I thought for some time about these two lines of code.

    shim_type = recipe['servable']['shim']
    x = importlib.import_module("home_run.{}".format(shim_type))(**recipe)
    return x
