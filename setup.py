from setuptools import find_packages, setup

setup(
    name="nnunet-path",  # change "src" folder name to your project name
    packages=find_packages(".", exclude=["tests*"]),
)
