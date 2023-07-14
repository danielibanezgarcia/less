import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="less",
    version="1.0.0",
    author="Daniel Ibanez",
    author_email="danielibanezgarcia@gmail.com",
    description="Code for the LeSS paper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danielibanezgarcia/less",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "spacy",
        "PyYAML>=5.3, <6.0",
    ],
    python_requires=">=3.6",
)
