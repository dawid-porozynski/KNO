from setuptools import setup, find_packages

setup(
    name="my-ml-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.16.2",
        "keras==3.10.0",
        "matplotlib==3.9.4",
        "numpy==1.26.4",
    ],
)