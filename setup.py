import os
import toml
from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "isaacsim-rl",
    "isaaclab",
    "isaaclab-tasks",
    "isaaclab-rl",
    "rsl-rl",
]

setup(
    name="humanoid_parkour",
    version="1.0.0",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.10",
    description="G1 Humanoid Parkour Locomotion - Course Project",
)
