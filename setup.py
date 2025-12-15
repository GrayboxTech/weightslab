import os
import pathlib
from setuptools import setup, find_packages


def get_requirements(file_path: pathlib.Path):
    """
    Read requirements.txt and optionally switch Torch variant between CPU and GPU.

    Control via environment variable:
    """
    def sanitize(dep: str) -> str:
        # Drop any inline pip flags (e.g., '--extra-index-url ...')
        if ' --' in dep:
            dep = dep.split(' ')[0]
        # Normalize torch build tags like 'torch==2.1.2+cpu' for install_requires
        if dep.startswith('torch==') and '+cpu' in dep:
            version = dep.split('==', 1)[1].split('+', 1)[0]
            dep = f'torch-cpu=={version}'
        if dep.startswith('torch==') and '+cu' in dep:
            # Remove build tag, keep base version for GPU default
            version = dep.split('==', 1)[1].split('+', 1)[0]
            dep = f'torch=={version}'
        return dep
    
    variant = os.getenv('WEIGHTSLAB_TORCH_VARIANT', '').lower().strip()
    with open(file_path, 'r', encoding='utf-8') as f:
        raw = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if variant:
        variant = variant.lower().strip()
    # unreachable due to early returns; kept for clarity
    # return raw
    if variant in {"cpu"}:
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
