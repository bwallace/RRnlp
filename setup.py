import os 
from setuptools import setup, find_packages

path_to_weights = os.path.join('rrnlp', 'models', 'weights')
path_to_minimap = os.path.join('rrnlp', 'models', 'util', 'minimap')

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = ["../requirements.txt", 
                os.path.join("../", path_to_weights, "weights_manifest.json")]
extra_files.extend(package_files(path_to_minimap))

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='rrnlp',
      version='1.0.3',
      description='NLP for EBM',
      url='https://github.com/bwallace/RRnlp',
      author='Byron Wallace, Iain Marshall',
      author_email='b.wallace@northeastern.edu',
      license='MIT',
      packages=find_packages(), 
      package_data={'': extra_files},
      install_requires=required,
      zip_safe=False)
