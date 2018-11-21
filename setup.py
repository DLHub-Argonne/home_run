from setuptools import setup, find_packages

setup(
    name='home_run',
    version='0.0.1',
    packages=find_packages(exclude=('test',)),
    install_requires=[
        "dlhub_sdk"
    ],
    python_requires=">=3.6",
    license="Apache License, Version 2.0",
    url="https://github.com/dlhub-argonne/home_run"
)
