#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Apply filter on faces",
    author="Phan Anh Duc",
    author_email="phananhduc138@gmail.com",
    url="https://github.com/PAD2003/apply_filter",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
    },
)
