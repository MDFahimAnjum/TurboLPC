from setuptools import setup, find_packages

setup(
    name='turbolpc',
    version='0.1',
    packages=find_packages(),
    author="Fahim Anjum",
    author_email="dr.fahim.anjum@gmail.com",
    description="TurboLPC is a fast, simple yet powerful Python library that provides the functionality of Linear Predictive Coding for signals.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MDFahimAnjum/TurboLPC",
    include_package_data=True,
    package_data={"": ["example_notebook.ipynb"]},
    python_requires='>=3.9',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=open('requirements.txt').readlines(),
)