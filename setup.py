from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup


with open("README.md", encoding="utf-8") as f:
    long_description = "\n" + f.read()


setup(
    name="efficientnet-lite-keras",
    version="1.0",
    description="A Keras implementation of EfficientNet Lite models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sebastian-sz/efficientnet-lite-keras",
    author="Sebastian Szymanski",
    author_email="mocart15@gmail.com",
    license="Apache",
    python_requires=">=3.6.0,<3.10",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    packages=["efficientnet_lite"],
)
