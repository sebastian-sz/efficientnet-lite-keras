from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup

with open("README.md", encoding="utf-8") as f:
    long_description = "\n" + f.read()


def _package_exists(name: str) -> bool:
    """Check whether package is present in the system."""
    try:
        get_distribution(name)
    except DistributionNotFound:
        return False
    else:
        return True


def _get_tensorflow_requirement():
    """Avoid re-download and misdetection of package."""
    lower = 2.2
    upper = 2.7

    if _package_exists("tensorflow-cpu"):
        return [f"tensorflow-cpu>={lower},<{upper}"]
    elif _package_exists("tensorflow-gpu"):
        return [f"tensorflow-gpu>={lower},<{upper}"]
    else:
        return [f"tensorflow>={lower},<{upper}"]


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
    install_requires=_get_tensorflow_requirement(),
)
