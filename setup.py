from setuptools import setup, find_packages

req = [p.strip() for p in open("requirements-minimal.txt").readlines()]

setup(
    name="boggler",
    version="1.0",
    long_description=__doc__,
    packages=find_packages(),
    entry_points={"console_scripts": ['boggler=boggler.cli:cli']},
    include_package_data=True,
    zip_safe=False,
    install_requires=req,
)
