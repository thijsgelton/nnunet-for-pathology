from setuptools import find_packages, setup

setup(
    name="nnunet_pathology",  # change "nnunet_pathology" folder name to your project name
    packages=find_packages(".", exclude=["tests*"]),
)
