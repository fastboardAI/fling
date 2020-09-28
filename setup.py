from setuptools import find_packages, setup
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fastboard-fling',
    version='0.0.1',
    maintainer='fastboardAI',
    author='Arnab Borah',
    license='MIT',
    description='fastLinguistics : unsupervised computational linguistics',
    long_description=long_description,
    long_description_content_type="text/markdown",
    download_url="https://github.com/fastboardAI/fling/dist",
    packages=find_packages(include=['fling'])
)
