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

addtional_mods = []
include_files=[
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\cilkrts20.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\ifdlg100.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\libchkp.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\libicaf.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\libifcoremd.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\libifcoremdd.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\libifcorert.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\libifcorertd.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\libifportmd.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\libimalloc.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\libiomp5md.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\libiompstubs5md.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\libmmd.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\libmmdd.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\libmpx.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\liboffload.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_avx.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_avx2.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_avx512.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_avx512_mic.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_core.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_def.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_intel_thread.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_mc.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_mc3.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_msg.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_rt.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_sequential.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_tbb_thread.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_vml_avx.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_vml_avx2.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_vml_avx512.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_vml_avx512_mic.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_vml_cmpt.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_vml_def.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_vml_mc.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_vml_mc2.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\mkl_vml_mc3.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\Library\\bin\\svml_dispmd.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\DLL\\tcl86t.dll",
    "C:\\Users\\Jonny\\AppData\\Local\\conda\\conda\\envs\\deeplearning\\DLL\\tk86t.dll"
]
disclude_mods=['Tkinter']
package_mods= ['numpy','matplotlib','tkinter']


executables = [
    Executable('test.py')
]

setup(name='tet',
      version='0.1',
      description='Sample cx_Freeze script',
      options = {'build_exe': {'includes': addtional_mods,'include_files': include_files ,'packages':package_mods,'excludes':disclude_mods }},
      executables=executables
      )
