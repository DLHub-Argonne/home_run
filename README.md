# DLHub home_run

`home_run` is a tool used by DLHub internally to turn a bunch of files and a recipe into an functional Python object. 

## Installation

`home_run` is not yet on PyPi. So, you have to install it by first cloning the repository and then calling `pip install -e .`

## Technical Details

The key ingredients for using `home_run` are files describing a function that will be served by DLHub.
These include a metadata file describing the servable (see 
[`dlhub_toolbox`](http://github.com/dlhub-argonne/dlhub_toolbox) for tools for creating these files, 
and [`dlhub_schemas`](http://github.com/dlhub-argonne/dlhub_schemas) for the schemas), and
the actual files that make up the servable (e.g., a TensorFlow model)

Each particular type of servable has its own recipe for going from these files to a Python object.
All recipes are a subclass of `BaseServable`, which provides the general framework for defining a servable object.
Each subclass has a match `BaseMetadataModel` class in `dlhub_toolbox`.
For example, the type of servable that can be described by the `PythonStaticMethodModel` can be run by the `PythonStaticMethodServable`.
   
