from setuptools import setup, find_packages

setup(
    name="chrono",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'flask',
        'numpy',
        'pandas',
        'torch',
        'transformers',
        'scipy',
        'scikit-learn'
    ]
) 