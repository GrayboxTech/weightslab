import os
import pathlib
from setuptools import setup, find_packages


def get_requirements(file_path: pathlib.Path):
    """
    Read requirements.txt and optionally switch Torch variant between CPU and GPU.

    Control via environment variable:
      - WEIGHTSLAB_TORCH_VARIANT=cpu | gpu (default: use file as-is)

    If 'cpu' is selected, any 'torch==' entries are converted to 'torch-cpu=='.
    If 'gpu' is selected, any 'torch-cpu==' entries are converted to 'torch=='.
    Comments and blank lines are ignored.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

    variant = os.environ.get('WEIGHTSLAB_TORCH_VARIANT')
    if variant:
        variant = variant.lower().strip()

    if variant in {"cpu", "gpu"}:
        converted = []
        for dep in raw:
            if variant == "cpu":
                # Prefer CPU-only torch wheel to avoid pulling NVIDIA CUDA deps
                if dep.startswith("torch") and "cpu" not in dep:
                    dep = dep + "+cpu --extra-index-url https://download.pytorch.org/whl/cpu"
            converted.append(dep)
        return converted

    # Default: return as-is
    return raw


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
