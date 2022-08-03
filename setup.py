import pathlib
from setuptools import setup,find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="test_APIMakeSens",
    version="1.1.7",
    description="test MakeSense API",
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
    include_package_data=True,
    install_requires=["pandas", "requests","datetime"]
    ,
    entry_points={
        "console_scripts": [
            "test-MakeSens-API=MakeSens.__main__:main",
        ]
    },
)