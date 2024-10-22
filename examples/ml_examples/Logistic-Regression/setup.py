# import setuptools
from setuptools import find_packages, setup
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
    name='LogisticRegressionProject',
    version='0.0.1',
    description='Build an affair classification model using Logistic Regression',
    author='Debabrata Ghorai, Ph.D.',
    author_email='ghoraideb@gmail.com',
    url='https://github.com/dghorai83',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages(),
    long_description=readme()
)
