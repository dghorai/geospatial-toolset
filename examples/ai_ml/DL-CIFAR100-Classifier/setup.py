import setuptools

__version__ = '0.0.1'

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

REPO_NAME = "cifar100classifier-deep-learning"
SRC_REPO = "CNNClassifier"
AUTHOR_NAME = "dghorai"
AUTHOR_EMAIL = "ghoraideb@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author="Debabrata Ghorai",
    author_email=AUTHOR_EMAIL,
    description="A Python package for CIFAR-100 dataset classifcation",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
