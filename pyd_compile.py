from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        ["Model.py", "MyDataset.py"],
        compiler_directives={"language_level": 3}  # 保留其他有效的選項
    )
)