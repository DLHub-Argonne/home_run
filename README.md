# DLHub home_run

[![Build Status](https://travis-ci.org/DLHub-Argonne/home_run.svg?branch=master)](https://travis-ci.org/DLHub-Argonne/home_run)[![Coverage Status](https://coveralls.io/repos/github/DLHub-Argonne/home_run/badge.svg?branch=master)](https://coveralls.io/github/DLHub-Argonne/home_run?branch=master)

`home_run` is a tool used by DLHub internally to turn a bunch of files and a recipe into an functional Python object. 

## Installation

`home_run` is not yet on PyPi. So, you have to install it by first cloning the repository and then calling `pip install -e .`

## Technical Details

The key ingredients for using `home_run` are files describing a function that will be served by DLHub.
These include a metadata file describing the servable (see 
[`dlhub_sdk`](http://github.com/dlhub-argonne/dlhub_sdk) for tools for creating these files, 
and [`dlhub_schemas`](http://github.com/dlhub-argonne/dlhub_schemas) for the schemas), and
the actual files that make up the servable (e.g., a Keras hdf5 file).

Each particular type of servable has its own recipe for going from these files to a Python object.
All recipes are a subclass of `BaseServable`, which provides the general framework for defining a servable object.
Each subclass has a matching `BaseMetadataModel` class in `dlhub_sdk`.
For example, the type of servable that can be described by the `PythonStaticMethodModel` can be run by the `PythonStaticMethodServable`.
   
