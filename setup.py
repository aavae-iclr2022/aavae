#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='src',
    version='0.0.0',
    description='Describe Your Cool Project',
    author='',
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/aavae-iclr2022/aavae.git',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)
