from logging.handlers import RotatingFileHandler
import importlib
import logging


def create_servable(recipe, debug=False):
    """Given a recipe, create a servable

    Args:
        recipe (dict): Recipe describing a servable to be created
        debug (bool): Whether to print DEBUG-level events to the log
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

    # Define the logging settings
    logger = logging.getLogger('home_run')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = RotatingFileHandler('dlhub.log', maxBytes=4 * 1024 * 1024, backupCount=1)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    # Get the shim type
    shim_type = recipe['servable']['shim']
    logger.info('Found shim type: ' + shim_type)

    # Get the module and class name for the shim
    module_name, class_name = shim_type.split(".")
    mod = importlib.import_module("home_run.{}".format(module_name))
    cls = getattr(mod, class_name)
    return cls(**recipe)
