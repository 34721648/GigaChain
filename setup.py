from setuptools import setup, find_packages

setup(
    name='gigachain',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
       'langchain>=0.0.264'
    ],
)