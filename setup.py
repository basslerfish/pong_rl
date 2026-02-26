"""
Allows installing as Python package.
"""
from setuptools import setup

setup(
    name="pong_rl",  # The name of your package
    version="0.1.0",         # Version number
    description="Use a DQN to play Pong",  # A short description
    packages=["pong_rl"],  # List of package directories
)