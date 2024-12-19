"""Archivo de configuración del modulo MakeSens"""
import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="APIMakeSens",
    version="1.4.10",
    description="MakeSense API",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/MakeSens-Data/MakeSensAPI_Python",
    author="MakeSens",
    author_email="makesens19@gmail.com",
    license="GNU General Public License v3.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    packages=["MakeSens"],
    package_data={
        'MakeSens': ['colors_by_variable.json'],
    },
    include_package_data=True,
    data_files=[('data', ['MakeSens/colors_by_variable.json'])],
    install_requires=["pandas", "requests","datetime"],
    entry_points={
        "console_scripts": [
            "test-MakeSens-API=MakeSens.__main__:main",
        ]
    },
)
# End-of-file