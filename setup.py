from setuptools import setup, find_packages
import os

# single source of truth for package version
version_ns = {}
with open(os.path.join("home_run", "version.py")) as f:
    exec(f.read(), version_ns)
version = version_ns['__version__']

setup(
    name='home_run',
    version=version,
    packages=find_packages(exclude=['tests', 'tests.*']),
    python_requires=">=3.6",
    install_requires=['requests'],
    description="Turns DLHub metadata into functional Python objects",
    long_description=open("README.md").read(),
    license="Apache License, Version 2.0",
    author="Ryan Chard",
    author_email="rchard@anl.gov",
    url="https://github.com/dlhub-argonne/home_run",
    keywords=[
        "DLHub",
        "Data and Learning Hub for Science",
        "machine learning",
        "data publication",
        "software publication",
        "reproducibility"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: System :: Distributed Computing"
    ],
)
