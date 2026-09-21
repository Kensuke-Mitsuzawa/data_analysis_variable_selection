[project]
name = "data_analysis_variable_selection"
version = "0.1"
description = ""
readme = "README.md"
authors = [
    { name = "Kensuke Mitsuzawa", email = "kensuke.mit@gmail.com" }
]
requires-python = ">=3.9"
dependencies = [
]

[project.optional-dependencies]
dask_visual = [
    "bokeh==2.4.2",
]
experiment_compare = [
    "future",
]

[dependency-groups]
dev = [
    "pytest",
    "pytest-resource-path>=1.4.1",
    "typeguard>=4.4.4",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["*"]