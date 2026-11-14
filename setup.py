# -*- coding: utf-8 -*-

from setuptools import find_packages
from distutils.core import setup

setup(
    name="sapai",
    version="0.1.0",
    packages=[
        "sapai",
    ],
    # find_packages(exclude=[]),
    install_requires=["numpy", "torch"],
    ### Game data built from data/turtle_pack.yaml by sapai.build_data
    package_data={"sapai": ["game_data.json"]},
    data_files=[],
)
