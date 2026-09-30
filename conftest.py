"""pytest 根 conftest。

存在的唯一目的：让 pytest 在收集测试时把**仓库根目录**加入 sys.path，
从而 `from judge import ...` 可用。

之所以不用 pytest.ini 的 `pythonpath` 选项：那需要写绝对路径，
会把本机目录结构泄露进版本库。这个空文件则不含任何路径信息。
"""
