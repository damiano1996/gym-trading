"""
Setup of the project.
"""

from setuptools import setup

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='gym_trading',
    version='0.0.1',
    install_requires=required_packages
)
