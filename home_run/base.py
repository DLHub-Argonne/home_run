from functools import wraps
from . import __version__


class BaseServable:
    """Base class for all servable objects.

    The base defines the methods needed to initialize a functional servable from its methods.
    These include:

    - "_run": The default method for all servables. It will be called by the DLHub workers, and
        must be implemented for each new type of servable.
    - "_build": A method that is invoked by the initializer and does any preparations needed to
        run all of the methods defined in the servable metadata. These could include loading
        serialized objects from disk, starting a external server process, etc. This method
        must also create the functions for all methods besides the "_run" function. These new
        methods are created by calling the "_set_function" method (detailed instructions
        are provided in the docstring for the "_set_function" method).

    """

    def __init__(self, datacite, dlhub, servable):
        """Initialize the class

        Args:
            datacite (dict): Metadata about provenance
            dlhub (dict): Metadata used by DLHub service
            servable (dict): Metadata describing the servable. Instructions on what it can perform
        """

        self.datacite = datacite
        self.dlhub = dlhub
        self.servable = servable

        # Call the build function
        self._build()

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

    def run(self, inputs, **parameters):
        """Invoke the main operation for this servable

        Args:
            inputs: Inputs to the function
            parameters (dict): Any options to use for the class. Overrides the defaults
        """

        # Get the parameters
        method_name = 'run'
        params = self._get_method_parameters(method_name, parameters)

        return self._run(inputs, **params)

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

    def _run(self, inputs, **parameters):
        """Private function to be implemented by subclass"""
        raise NotImplementedError()

    def _set_function(self, method_name, function):
        """Define a new method for this class

        Creates a new method for this servable object, given the name of the desired method
        and a function object. The name of the method should match one of the ones defined in the
        servable metadata. The function must take two arguments: the inputs to the servable function
        and any configurable parameters as a dictionary (same as "_run")

        Args:
            method_name (string): Name of the method
            function (function pointer): Function to set
        """

        def new_function(inputs, **parameters):
            params = self._get_method_parameters(method_name, parameters)
            return function(inputs, **params)

        setattr(self, method_name, new_function)

    @staticmethod
    def get_version():
        """Get the version of home_run used by this servable

        Intended for debugging purposes

        Returns:
            (string) Version of home_run
            """
        return __version__
