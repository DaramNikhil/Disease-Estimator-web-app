from setuptools import setup, find_packages
# setup

setup(

    name="human disease prediction",
    description="human disease prediction project",
    author="Nikhil Daram",
    author_email="nikhildaram51@gmail.com",
    install_requires=[
        "pandas ",
        "numpy"
    ],
    classifiers=[

        "programming languages::python3"
    ],
    packages=find_packages()
)
