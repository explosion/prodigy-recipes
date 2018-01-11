#!/usr/bin/env python
# coding: utf8
#
# USAGE: python package.py

from __future__ import unicode_literals

import shutil, sys, glob, os
from pathlib import Path
from setuptools import setup, find_packages
from Cython.Build import cythonize
import wheel.bdist_wheel
from setuptools.command.build_ext import build_ext as base_build_ext

if len(sys.argv) == 1:
    sys.argv.append('build_ext')


def rename_py_pyx(root, package):
    bld_dir = root / 'cython_src' / package
    if not bld_dir.exists():
        shutil.copytree(str(root / package), str(bld_dir))
    keep_python = ('__init__.py', '__main__.py', 'app.py', 'about.py')
    for path, subdirs, files in os.walk(str(bld_dir)):
        if 'recipes' in path:
            continue
        for name in files:
            if name.endswith('.py') and name not in keep_python:
                pyx = name.replace('.py', '.pyx')
                os.rename(os.path.join(path, name), os.path.join(path, pyx))
    packages = find_packages('cython_src', exclude=['__pycache__'])
    return bld_dir, packages


def setup_package():
    package_name = 'prodigy'
    root = Path(__file__).parent.resolve()
    bld_dir, packages = rename_py_pyx(root, package_name)
    print(packages)

    # Read in package meta from about.py
    about_path = root / package_name / 'about.py'
    with about_path.open('r', encoding='utf8') as f:
        about = {}
        exec(f.read(), about)

    # Read in requirements and split into packages and URLs
    requirements_path = root / 'requirements.txt'
    with requirements_path.open('r', encoding='utf8') as f:
        requirements = [line.strip() for line in f]

    # Create home directory and empty config JSON file if it doesn't exist
    config_dir = Path.home() / '.{}'.format(package_name)
    if not config_dir.exists():
        config_dir.mkdir()
        config_file = config_dir / '{}.json'.format(package_name)
        config_file.open('w', encoding='utf8').write('{\n\n}')

    setup(
        name=package_name,
        description=about['__summary__'],
        author=about['__author__'],
        author_email=about['__email__'],
        url=about['__uri__'],
        version=about['__version__'],
        license=about['__license__'],
        packages=packages,
        ext_modules=cythonize('cython_src/**/*.pyx'),
        package_data={package_name: ['static/*', 'static/fonts/*'],
                      '': ['*.so']},
        install_requires=requirements,
        zip_safe=False,
        package_dir={"": bld_dir.parts[-2]},
        scripts=['bin/prodigy', 'bin/pgy'],
    )

if __name__ == '__main__':
    setup_package()
