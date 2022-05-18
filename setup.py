from setuptools import find_packages, setup

setup(
    name="src",  # change "src" folder name to your project name
    packages=find_packages(".", exclude=["tests*"]),
)
