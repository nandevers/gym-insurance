from setuptools import setup, find_packages

setup(
    name="gym_insurance",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "benfordslaw",
        "gym",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "tensorboard",
        "tensorflow",
    ],
    author="Fernando Barbosa",
    author_email="f191114@dac.unicamp.com",
    description="A python package for generating random numbers following Benford's distribution",
    keywords="benford random distribution",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
