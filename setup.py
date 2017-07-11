# -*- coding: utf-8 -*-

# A very simple setup script to create a single executable
#
# hello.py is a very simple 'Hello, world' type script which also displays the
# environment in which the script runs
# Run the build process by running the command 'python setup.py build'
#
# If everything works well you should find a subdirectory in the build
# subdirectory that contains the files needed to run the script without Python

from cx_Freeze import setup, Executable
import os
os.environ['TCL_LIBRARY'] = "C:\\ProgramData\\Miniconda3\\tcl\\tcl8.6"
os.environ['TK_LIBRARY'] = "C:\\ProgramData\\Miniconda3\\tcl\\tk8.6"

addtional_mods = ['numpy.core._methods', 'numpy.lib.format', 'google']

executables = [
    Executable('main.py')
]

setup(name='tet',
      version='0.1',
      description='Sample cx_Freeze script',
      options = {'build_exe': {'includes': addtional_mods}},
      executables=executables
      )
