from setuptools import setup, find_packages

setup(
    name="sgminigrid",
    version="0.0.1",
    description='MiniGrid with subtasks',
    author='Anthony Liu',
    license='MIT',
    packages=['sgminigrid'],
    install_requires=[
        'minigrid'
    ],
    python_requires='>=3.7',
    entry_points={
        "gymnasium.envs": ["__root__ = sgminigrid.__init__:register_sgminigrid_envs"]
    },
)
