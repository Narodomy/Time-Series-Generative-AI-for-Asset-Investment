from setuptools import setup, find_packages

setup(
    name="ts_genai_inv",
    version="0.1",
    packages=find_packages(where="src"),  # Find the package in src/
    package_dir={"": "src"},              # Said where the root package at src/
)