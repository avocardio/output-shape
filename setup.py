from setuptools import setup

setup(
  name='output-shape',
  version='0.0.1',
  description='A very lightweight and minimalistic output shape examiner of layers and models.',
  long_description=open('README.md').read()[:2000],
  author='avocardio',
  packages=['output_shape'],
  license='MIT',
  url = 'https://github.com/avocardio/output-shape',
  long_description_content_type = 'text/markdown',
  install_requires=[
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)