from setuptools import setup, find_packages
import irp

setup(
    name="irp",
    version=irp.__version__,
    packages=find_packages(),

    # source code layout
    test_suite="irp.test.test_suite",

    # Generating the command-line tool
    entry_points={
        "console_scripts": [
        ]
    },

    # author and license
    author="David Tolpin",
    author_email="david.tolpin@gmail.com",
    description="Case studies for Intrusions in Renewal Processes",

    # dependencies
    install_requires=[],
    dependency_links=[]
)
