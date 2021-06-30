import os 
from setuptools import setup, find_packages

path_to_weights = os.path.join('rrnlp', 'models', 'weights')

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files(path_to_weights)

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='rrnlp',
      version='0.1',
      description='NLP for EBM',
      url='https://github.com/bwallace/RRnlp',
      author='Byron Wallace, Iain Marshall',
      author_email='b.wallace@northeastern.edu',
      license='MIT',
      packages=find_packages(), #['rrnlp', 'rrnlp.models'],
      package_data={'': extra_files},
      install_requires=required,
      zip_safe=False)