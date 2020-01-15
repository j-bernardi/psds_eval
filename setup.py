"""
setup.py for the psds package
"""
import os
import setuptools


def package_file(fname):
    return os.path.relpath(os.path.join(os.path.dirname(__file__), fname))


__version__ = None
# Get the version number from the version file
with open(package_file("version.py")) as fp:
    exec(fp.read())
assert __version__ is not None


setuptools.setup(
    name="psds_eval",
    version=__version__,
    author="Audio Analytic Ltd.",
    maintainer="Audio Analytic Ltd.",
    description="A module to calculate Polyphonic Sound Detection Score",
    license="MIT",
    keywords="polyphonic sound detection evaluation score",
    long_description=open(package_file("README.md")).read(),
    long_description_content_type='text/markdown',
    url="https://github.com/audioanalytic/psds_eval",
    python_requires=">=3.6",
    package_dir={"": package_file("src")},
    packages=setuptools.find_packages(package_file("src")),
    install_requires=["pandas>=0.19",
                      "numpy>=1.9.0",
                      "matplotlib>=3.1.0",
                      "pytest>=4.3"]
)
