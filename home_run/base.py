from . import __version__


class BaseServable:
    """Base class for all servable objects"""

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
        params = dict(self.servable['methods']['run'].get('parameters', {}))
        params.update(parameters)

        return self._run(inputs, **params)

    def _run(self, inputs, **parameters):
        """Private function to be implemented by subclass"""
        raise NotImplementedError()

    @staticmethod
    def get_version():
        """Get the version of home_run used by this servable

        Intended for debugging purposes

        Returns:
            (string) Version of home_run
            """
        return __version__
