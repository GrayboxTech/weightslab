import os
import pathlib
from setuptools import setup, find_packages


def get_requirements(file_path: pathlib.Path):
    """
    Read requirements.txt and optionally switch Torch variant between CPU and GPU.

    Control via environment variable:
      - WEIGHTSLAB_TORCH_VARIANT=cpu | gpu (optional)

    Notes:
      - install_requires must contain valid specifiers only; no pip flags allowed.
      - For CPU variant, transform 'torch==X' to 'torch-cpu==X'.
      - Sanitize versions like '2.1.2+cpu', '2.1.2-cpu', '2.1.2+cu121' to base '2.1.2'.
    """

    variant = os.getenv('WEIGHTSLAB_TORCH_VARIANT', '').lower().strip()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

    def sanitize_line(line: str) -> str:
        # Drop any inline pip flags (e.g., '--extra-index-url ...')
        if ' --' in line:
            line = line.split(' ')[0]
        return line

    def normalize_torch(dep: str) -> str:
        if not (dep.startswith('torch')):
            return dep
        name, ver = dep.split('==', 1)
        # Strip build/variant suffixes from version (PEP 440 compliant for install_requires)
        base_ver = ver
        for sep in ('+cpu', '-cpu'):
            if sep in base_ver:
                base_ver = base_ver.split(sep, 1)[0]
        # strip CUDA tags like '+cu121' or '-cu121'
        if '+' in base_ver:
            base_ver = base_ver.split('+', 1)[0]
        if '-cu' in base_ver:
            base_ver = base_ver.split('-cu', 1)[0]

        # Select package name based on requested variant
        if variant == 'cpu':
            base_ver += '+cpu'
        # else keep the original name from requirements.txt

        return f'{name}=={base_ver}'

    deps = []
    for dep in raw:
        dep = sanitize_line(dep)
        dep = normalize_torch(dep)
        deps.append(dep)

    return deps

requirements = get_requirements(
    os.path.join(pathlib.Path(__file__).parent,
                'requirements.txt')
)

# Setup packages
setup(
    name='weightslab',
    version='0.0.0',
    description='Paving the way between black-box and white-box modeling.',
    url='https://github.com/GrayboxTech/weightslab',
    author='Alexandru-Andrei Rotaru',
    author_email='alexandru@graybx.com',
    license='BSD 2-clause',
    install_requires=requirements,
    packages=find_packages(include=['weightslab', 'weightslab.*']),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.11',
    ],
)
