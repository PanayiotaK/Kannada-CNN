from setuptools import setup, find_packages

setup(
  name = 'cnn_panagiota',  
  version = '0.0.1',  
  description = 'CNN for Kannada MNIST - Pytorch',  
  long_description_content_type="text/markdown",
  long_description='PyTorch implementation of a CNN for the Kannada MNIST challenge on Kaggle',
  author = 'Panagiota Konstantinou',
  author_email = 'constantinoupana@gmail.com',
  url = 'https://github.com/PanayiotaK/Kannada-CNN',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'CNN',
    'Kannada'
  ],
  packages=find_packages(),
  install_requires=[
    'matplotlib',
    'numpy',
    'pandas',
    'torch>=1.10',
    'torchvision',
    'scikit-learn',
    'csv'
  ],
  classifiers=[
    'Development Status :: 1 - Planning',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    "Operating System :: Unix",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft :: Windows",
    'Programming Language :: Python :: 3.10',
  ],
)