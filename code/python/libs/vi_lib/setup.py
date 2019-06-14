from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='lib',
    version='0.1.0',
    description='',  
    long_description="",
    url='https://bitbucket.org/aaronpmishkin/vi_lib/',
    author='',
    author_email='',
    classifiers=[
    ],
    keywords='',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'other']),
    install_requires=['torch', 'artemis-ml'],
    project_urls={
    },
)
