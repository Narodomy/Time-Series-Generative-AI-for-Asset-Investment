from setuptools import setup, find_packages

setup(
    name    = "tsgenai",
    version = "0.2.0",
    authors = [
        { name = "Narodom Yatnimit", email = "narodomy@outlook.com"},
    ],
    description  = "Time Series Generative AI for Asset Investment. TAIST Science Tokyo Thesis.",
    readme      = "README.md",
    license     = "MIT",
    license-files = ["LICEN[CS]E*"],
    
    [project.urls]
    Homepage = "https://github.com/Narodomy/Time-Series-Generative-AI-for-Asset-Investment",
    Issues = "https://github.com/Narodomy/Time-Series-Generative-AI-for-Asset-Investment/issues",

    requires-python = ">=3.11.0"
    packages=find_packages(where="src"),  # Find the package in src/
    package_dir={"": "src"},              # Said where the root package at src/
)