from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(
    name='destination_based_sales',
    version="1.0",
    description="Python package accompanying my work on the adjustment of country-by-country revenue variables.",
    packages=find_packages(),
    test_suite='tests',
    zip_safe=False
)
