# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='geometric-ifc',
    version='0.1.0',
    description='use some geometries',
    long_description=readme,
#    author='Fiona Collins',
#    author_email='',
#    url='',
#    license=license,
#    packages=find_packages(exclude=('tests', 'docs'))
)

