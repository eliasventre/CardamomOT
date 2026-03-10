"""
Setup configuration for CARDAMOM package.

CARDAMOM: Calibration And Regularized Dynamics And Mechanistic Optimization Method
A gene regulatory network inference method for time-course scRNA-seq datasets.
"""

from setuptools import setup, find_packages
import os

# Read the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='CardamomOT',
    version='2.0.0',
    description='Gene regulatory network inference from single-cell scRNA-seq data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Elias Ventre',
    url='https://github.com/yourusername/CardaSC',
    project_urls={
        'Bug Tracker': 'https://github.com/yourusername/CardaSC/issues',
        'Documentation': 'https://github.com/yourusername/CardaSC',
        'Source Code': 'https://github.com/yourusername/CardaSC',
    },
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'cardamom=CardamomOT.cli:main',
        ],
    },
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.19',
        'scipy>=1.5',
        'numba>=0.51',
        'anndata>=0.7',
        'scikit-learn>=0.23',
        'matplotlib>=3.3',
        'pandas>=1.1',
        'umap-learn>=0.5',
        'alive-progress>=2.0',
        # 'harissa==3.0.7',  # Temporarily disabled due to build issues
        'torch>=1.9',
        'torchdiffeq>=0.2',
        'joblib>=1.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.12',
            'black>=21.0',
            'flake8>=3.9',
            'isort>=5.9',
            'sphinx>=4.0',
        ],
        'notebooks': [
            'jupyter>=1.0',
            'notebook>=6.0',
            'ipython>=7.0',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='single-cell, gene regulatory network, causal inference, optimal transport',
)
