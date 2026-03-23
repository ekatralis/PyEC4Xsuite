# setup.py

from setuptools import setup
from pathlib import Path


version_file = Path(__file__).parent / "PyEC4XS/_version.py"
dd = {}
with open(version_file.absolute(), "r") as fp:
    exec(fp.read(), dd)
__version__ = dd["__version__"]


setup(
    name="PyEC4XS",
    version=__version__,
    packages=["PyEC4XS"],
    include_package_data=True,
)