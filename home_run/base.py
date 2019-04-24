from .version import __version__
from tempfile import TemporaryDirectory
from urllib3.util import parse_url
from requests import get
import logging
import os
logger = logging.getLogger(__name__)


# Chunk size for donwloading files
chunk_size = 1024 ** 3  # 1 MB chunks per file (TODO (lw): Is this a good size))


def _get_file(file_data: dict, tmpdir: str):
    """Ensure that a file is accessible at a local path

    Args:
        file_data (dict): Data about the file to be downloaded
    Returns:
        (str): Path to the file on the local system
    """

    # If the url is a known path, skip parsing
    if os.path.isfile(file_data['url']):
        return file_data['url']

    # Parse the URL
    parsed_url = parse_url(file_data['url'])

    if parsed_url.scheme == 'file' or parsed_url.scheme is None:
        logger.debug('Using local file {}'.format(parsed_url))
        return parsed_url.path
    elif parsed_url.scheme == 'http' or parsed_url.scheme == 'https':
        # Download the file via HTTP
        new_path = os.path.join(tmpdir, str(parsed_url.host), *parsed_url.path.split('/'))
        logger.debug('Downloading file from {} to {}'.format(file_data['url'], new_path))
        os.makedirs(os.path.dirname(new_path))
        req = get(file_data['url'], headers=file_data.get('headers', {}), stream=True)
        with open(new_path, 'wb') as fp:
            for chunk in req.iter_content(chunk_size):
                fp.write(chunk)
        return new_path
    else:
        raise NotImplementedError('Scheme not supported: {}'.format(parsed_url.scheme))


class BaseServable:
    """Base class for all servable objects.

    ## Invoking Operations the Shim ##

    Each shim is initialized by passing the servable metadata to the initializer,
    which will create the methods defined in the servable metadata.
    For example, a servable with methods named "run" and "train" will have two class methods,
    "run" and "train," after the shim is initialized.

    The signatures to these functions will always have three arguments:

    - ``inputs``: The data passed to the functions as inputs
    - ``files``: Data needed to access any files included in the inputs.
    Each file is defined by a dictionary with one mandatory key 'url',
    which stores the URL of the file (e.g., an HTTP address),
    and - optionally - 'headers', which includes any
    authorization headers needed to access the files.
    How this information must be passed to the servable varies depending on the
    arguments to the function:
        - If the function has a single file as the argument, a file dictionary is expected.
        - If the function has a list of files as the arguments,
        a list of file dictionaries is expected.
        - If the function has multiple inputs (i.e., it takes a tuple), then the function expects
        a list of inputs where members of the list corresponding to arguments with files
        are populated with a file dictionary or list of dictionaries if the arguments
        are single files or tuples. For example, if a function takes a file, a list of files and
        a float as inputs then the input for ``files`` is a list where element 0 is a file
        dictionary and element 1 is a list of file dictionaries.
    - ``parameters``: A dictionary of the keyword arguments to the function.

    The shim will manage downloading any required files, if necessary, and
    merging any user-defined parameters with the defaults provided in the
    servable metadata.

    ## Implementing a New Shim ##

    The base defines the methods needed to initialize a functional servable from its methods.
    These include:

    - ``_run`` The default method for all servables. It will be called by the DLHub workers, and
        must be implemented for each new type of servable.
    - ``_build``: A method that is invoked by the initializer and does any preparations needed to
        run all of the methods defined in the servable metadata. These could include loading
        serialized objects from disk, starting a external server process, etc. This method
        must also create the functions for all methods besides the "_run" function. These new
        methods are created by calling the "_set_function" method (detailed instructions
        are provided in the docstring for the "_set_function" method).

    When adding logging events, use INFO level or higher for status messages during servable
    construction (as this only happens once) and DEBUG-level for events that occur at each
    invocation. WARN level or higher can occur whenever necessary.


    """

    def __init__(self, datacite, dlhub, servable):
        """Initialize the class

        Args:
            datacite (dict): Metadata about provenance
            dlhub (dict): Metadata used by DLHub service
            servable (dict): Metadata describing the servable. Instructions on what it can perform
        """

        logger.info('Creating a servable for {} version {}'.format(dlhub['name'],
                                                                   dlhub.get('id', None)))
        self.datacite = datacite
        self.dlhub = dlhub
        self.servable = servable

        # Call the build function
        self._build()

        # Create the `run` method
        self._set_function('run', self._run)

    def _build(self):
        """Add new functions to this class, as specified in the servable metadata"""
        raise NotImplementedError()

    def get_recipe(self):
        """Return the recipe used to create this servable.

        Intended for debugging purposes

        Returns:
            (dict) Recipe used to create the object
        """
        return {'datacite': self.datacite, 'dlhub': self.dlhub, 'servable': self.servable}

    def _get_method_parameters(self, method_name, parameters):
        """Get the parameters for a method by combining the user-supplied parameters with the
        defaults set in the servable definition

        Args:
            method_name (string): Name of the method
            parameters (dict): User-supplied parameters
        Returns:
            (dict) User parameters updated with the defaults
        """
        params = dict(self.servable['methods'][method_name].get('parameters', {}))
        params.update(parameters)
        return params

    def _get_files(self, method_name, inputs, files, tmpdir):
        """Generate local paths for file inputs

        Args:
            method_name (str): Name of the method to be invoked
            inputs: Input arguments to the function
            files (list or dict): List of files to be accessed
            tmpdir (str): Path to copy any remote files to
        Returns:
             Inputs substituted to handle the local files
        """

        # Get the input argument description
        method_inputs = self.servable['methods'][method_name]['input']

        # Case 1: Single file as input
        if method_inputs['type'] == 'file':
            return _get_file(files, tmpdir)
        elif method_inputs['type'] == 'list' and method_inputs['item_type'] == {'type': 'file'}:
            return [_get_file(f, tmpdir) for f in files]
        elif method_inputs['type'] == 'tuple':
            new_inputs = list(inputs)
            for i, (arg_files, arg_type) in enumerate(zip(files, method_inputs['element_types'])):
                if arg_type['type'] == 'file':
                    new_inputs[i] = _get_file(arg_files, tmpdir)
                elif arg_type['type'] == 'list' and arg_type['item_type'] == {'type': 'file'}:
                    new_inputs[i] = [_get_file(f, tmpdir) for f in arg_files]
            return new_inputs
        else:
            raise ValueError('Input type not valid for file inputs')

    def _run(self, inputs, **parameters):
        """Private function to be implemented by subclass"""
        raise NotImplementedError()

    def _set_function(self, method_name, f):
        """Define a new method for this class

        Creates a new method for this servable object, given the name of the desired method
        and a function object. The name of the method should match one of the ones defined in the
        servable metadata. The function must take two arguments: the inputs to the servable function
        and any configurable parameters as a dictionary (same as "_run")

        The purpose of the _set_function method is to avoid the need
        for functions in different shims to need to implement the code
        for downloading files locally and handling default arguments.

        Args:
            method_name (string): Name of the method
            f (function pointer): Function to set
        """

        def new_function(inputs, files=None, parameters=None):
            # Get the default parameters
            if parameters is None:
                parameters = dict()
            params = self._get_method_parameters(method_name, parameters)
            logger.debug('Running method {} with params: {}'.format(method_name, params))

            if files is None:
                return f(inputs, **params)
            else:
                # Download files, if needed
                with TemporaryDirectory() as td:
                    new_inputs = self._get_files(method_name, inputs, files, td)
                    return f(new_inputs, **params)

        setattr(self, method_name, new_function)
        logger.info('Added function to servable {}: {}'.format(self.dlhub['name'],
                                                               method_name))

    @staticmethod
    def get_version():
        """Get the version of home_run used by this servable

        Intended for debugging purposes

        Returns:
            (string) Version of home_run
        """
        return __version__
