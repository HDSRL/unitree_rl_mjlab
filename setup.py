"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup

INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
    "pandas>=2.0",
    "pyarrow>=14.0",
    "plotly>=5.18",
]

setup(
    name="unitree_rl_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)