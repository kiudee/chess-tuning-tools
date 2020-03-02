#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "numpy>=1.18",
    "scipy>=1.3.2",
    "pytz",
    "joblib",
    "scikit-optimize",
    "emcee>=3.0.2",
    "psycopg2",
    "bask>=0.1.0",
    "sqlalchemy>=1.3",
    "pandas>=1.0.1"
]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest>=3"]

setup(
    author="Karlson Pfannschmidt",
    author_email="kiudee@mail.upb.de",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A collection of tools for local and distributed tuning of chess engines.",
    entry_points={"console_scripts": ["tune=tune.cli:cli"]},
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="chess, tuning, optimization, engine",
    name="chess-tuning-tools",
    packages=find_packages(include=["tune", "tune.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/kiudee/chess-tuning-tools",
    version="0.3.0",
    zip_safe=False,
)
