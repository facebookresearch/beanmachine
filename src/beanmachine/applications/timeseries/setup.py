from setuptools import find_packages, setup

setup(
    name="timeseries",
    version="0.0.0",
    description="Time series modeling using GPs and Bayesian Structural Time Series",
    packages=find_packages(include=["sts", "sts.*"]),
    install_requires=["gpytorch", "torch", "numpy", "pandas"],
    extras_require={
        "test": [
            "flake8",
            "pytest>=4.1",
        ],
        "examples": ["matplotlib", "seaborn"],
    },
)
