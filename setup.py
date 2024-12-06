import setuptools


with open("README.md", "rb") as fh:
    long_description = fh.read().decode()
with open("diffrp/version.py", "r") as fh:
    exec(fh.read())
    __version__: str


def packages():
    return setuptools.find_packages(include=['diffr*'])


setuptools.setup(
    name="diffrp",
    version=__version__,
    author="flandre.info",
    author_email="flandre@scarletx.cn",
    description="Differentiable Render Pipelines with PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eliphatfs/diffrp",
    packages=packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'torch',
        'pyexr',
        'calibur',
        'trimesh[easy]',
        'torch_redstone',
        'typing_extensions',
        'nvdiffrast @ git+https://github.com/eliphatfs/nvdiffrast.git',
    ],
    package_data={"diffrp.plugins": ["**/*.c", "**/*.h"], "diffrp": ["resources/**/*.*"]},
    exclude_package_data={
        'diffrp': ["*.pyc"],
    },
    python_requires='~=3.7'
)
