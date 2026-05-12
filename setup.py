# -*- coding: utf-8 -*-
"""
Created on Fri Apr 5 19:26:11 2024

@author: Debabrata Ghorai, Ph.D.

This script used to configure, build, and distribute Python packages.

"""

from setuptools import setup, find_packages
from typing import List


def readme():
    with open('README.md') as f:
        return f.read()


def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    return requirements


setup(
    name='geospatial-toolset',
    version='0.1.0',
    description='Geospatial Utilities for Remote Sensing and GIS Application',
    author='Debabrata Ghorai, Ph.D.',
    author_email='ghoraideb@gmail.com',
    url='https://github.com/dghorai',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages(),
    license='MIT License',    
    long_description=readme()
)
