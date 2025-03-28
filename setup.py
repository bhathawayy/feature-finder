import io
import os

from setuptools import (find_packages, setup)

# Define metadata
NAME = "featureFinder"
DESCRIPTION = "Simple GUI for feature detection."
URL = ""
EMAIL = "bhathawayy@yahoo.com"
AUTHOR = "Brooke Hathaway"
REQUIRES_PYTHON = ">=3.6"
VERSION = "1.1.0"

# Get current path
here = os.path.abspath(os.path.dirname(__file__))

# Required or optional package dependencies
required_modules = []
r_path = os.path.join(here, "requirements.txt")
if os.path.exists(r_path):
    with open(r_path, "r") as r_file:
        required_modules = r_file.read().splitlines()

# Get description from the "readme" file
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the versioning
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

# Call main setup command
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_data={'featureFinder': ['data/*', 'resources/*']},
    install_requires=required_modules,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy"
    ]
)
