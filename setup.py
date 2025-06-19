import os
import re

import setuptools


def get_requirements(req_path: str):
    with open(req_path, encoding='utf8') as f:
        return f.read().splitlines()


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_dir, 'README.md'), encoding='utf-8') as f:
        return f.read()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, 'voicehub', '__init__.py')
    with open(version_file, encoding='utf-8') as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


INSTALL_REQUIRES = get_requirements("requirements.txt")

setuptools.setup(
    name='voicehub',
    version=get_version(),
    author="kadirnardev",
    author_email='kadir.nar@hotmail.com',
    license="Apache-2.0",
    description="VoiceHub: A Unified Inference Interface for TTS Models",
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/kadirnar/voicehub',
    install_requires=INSTALL_REQUIRES,
    packages=setuptools.find_packages(),
)
