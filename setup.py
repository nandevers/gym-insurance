from setuptools import setup, find_packages

setup(
    name='benford-random',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A python package for generating random numbers following Benford\'s distribution',
    keywords='benford random distribution',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
