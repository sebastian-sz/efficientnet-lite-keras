from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup

IGNORE_REQUIREMENTS = ("pre-commit",)


with open("README.md", encoding="utf-8") as f:
    long_description = "\n" + f.read()


def _fix_tf_requirement_if_other_installed(requirements):
    tf_requirement_idx = [
        i for i, x in enumerate(requirements) if x.startswith("tensorflow>=")
    ][0]
    tf_requirement = _modify_tensorflow_requirement(
        requirements.pop(tf_requirement_idx)
    )
    requirements.append(tf_requirement)
    return requirements


def _modify_tensorflow_requirement(tf_requirement):
    assert tf_requirement.startswith("tensorflow>=")
    if _package_exists("tensorflow-cpu"):
        tf_requirement = tf_requirement.replace("tensorflow", "tensorflow-cpu")
    elif _package_exists("tensorflow-gpu"):
        tf_requirement = tf_requirement.replace("tensorflow", "tensorflow-gpu")
    return tf_requirement


def _package_exists(name: str) -> bool:
    """Check whether package is present in the system."""
    try:
        get_distribution(name)
    except DistributionNotFound:
        return False
    else:
        return True


with open("requirements.txt", "r") as f:
    content = f.readlines()
    requirements = [
        line
        for line in content
        if not line.startswith(("#", "\n", *IGNORE_REQUIREMENTS))
    ]
    requirements = _fix_tf_requirement_if_other_installed(requirements)


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
    install_requires=requirements,
)
