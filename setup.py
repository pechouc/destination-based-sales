from setuptools import setup, find_packages

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(
    name='destination_based_sales',
    version="1.0",
    description="Python package accompanying my work on the adjustment of country-by-country revenue variables.",
    packages=find_packages(),
    include_package_data=True,
    test_suite='tests',
    zip_safe=False
)
