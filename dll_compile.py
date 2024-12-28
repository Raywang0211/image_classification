from setuptools import setup, Extension
from Cython.Build import cythonize

# 定義需要編譯的目標文件
extensions = [
    Extension("AImodelInference", ["AImodelInference.py"]),       # 將 model.py 編譯成 model.dll
    Extension("MyDataset", ["MyDataset.py"]),  # 將 dataset.py 編譯成 dataset.dll
]

# 配置打包
setup(
    name="Myinference",
    ext_modules=cythonize(
        extensions, 
        compiler_directives={"language_level": "3"}  # 設置 Python 3 語言層級
    )
)