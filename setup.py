from setuptools import setup, find_packages

setup(
    name="tsgenai",
    version="0.1.4",
    packages=find_packages(where="src"),  # Find the package in src/
    package_dir={"": "src"},              # Said where the root package at src/
)