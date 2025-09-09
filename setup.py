"""
Setup configuration for the Conductor PyTorch Backend Integration.

This setup.py provides compatibility with older build systems while
pyproject.toml is the primary configuration for modern Python packaging.
"""

from setuptools import setup, find_packages
import os


# Read version from package
def get_version():
    """Extract version from package __init__.py"""
    version_file = os.path.join(os.path.dirname(__file__), "conductor", "__init__.py")
    with open(version_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip("\"'")
    return "0.1.0"


# Read long description from README
def get_long_description():
    """Read long description from README.md if available"""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Conductor: PyTorch Backend Integration for GCU Hardware"


setup(
    name="conductor",
    version=get_version(),
    author="Conductor Team",
    author_email="conductor@example.com",
    description="PyTorch Backend Integration for GCU Hardware",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/conductor/conductor-pytorch",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=0.900",
            "flake8>=4.0",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "performance": [
            "pybind11>=2.6",  # For optional C++ extensions
        ],
    },
    entry_points={
        "console_scripts": [
            "conductor-info=conductor.utils.info:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="pytorch, compiler, backend, gcu, machine-learning, deep-learning",
)
