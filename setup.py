import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mmxai",
    version="0.0.1",
    author="Jay Jiang, Yongkang Zhao, Zhi Wang, Bojia Mao, Genze Jiang, Chenyu Zhang",
    author_email="junqi.jiang20@imperial.ac.uk, yongkang.zhao20@imperial.ac.uk, \
            zhi.wang18@imperial.ac.uk, bojia.mao16@imperial.ac.uk, \
            genze.jiang20@imperial.ac.uk, chenyu.zhang16@imperial.ac.uk",
    description="MSc Computing Group Project - An Explainable AI Interface for \
            Multimodal Classification Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.doc.ic.ac.uk/g207004202/explainable-multimodal-classification",
    project_urls={
        "Bug Tracker": "https://gitlab.doc.ic.ac.uk/g207004202/explainable-multimodal-classification/-/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
)
