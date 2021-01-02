import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Plague_App",
    version = "1.0.0",
    author="Daisymay55",
    description="an app that simulates plague outbreaks where the user can modify variables as required",
    url="https://github.com/Daisymay55/LeedsUni-Coursework",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
