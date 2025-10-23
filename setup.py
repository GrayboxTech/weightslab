import os
import pathlib
from setuptools import setup, find_packages


def get_requirements(file_path: pathlib.Path):
    """
    Reads the contents of the requirements.txt file and returns a clean list
    of dependencies (ignoring comments and blank lines).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Filter out empty lines and comments
            return [
                line.strip()
                for line in f
                if line.strip() and not line.startswith('#')
            ]
    except FileNotFoundError:
        # Return an empty list if the file does not exist
        return []


# Setup packages
setup(
    name='weightslab',
    version='0.0.0',
    description='Paving the way between black-box and white-box modeling.',
    url='https://github.com/GrayboxTech/weightslab',
    author='Alexandru-Andrei Rotaru',
    author_email='alexandru@graybx.com',
    license='BSD 2-clause',
    install_requires=get_requirements(
        os.path.join(pathlib.Path(__file__).parent,
                     'requirements.txt')
    ),
    packages=find_packages(include=['weightslab', 'weightslab.*']),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.11',
    ],
)
