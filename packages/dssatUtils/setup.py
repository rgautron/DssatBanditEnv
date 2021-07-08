import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dssatUtils",
    version="0.0.1",
    author="Romain Gautron",
    author_email="r.gautron@cgiar.org",
    description="Utilitary functions to couple DSSAT and Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.0',
    license='Proprietary',
)
