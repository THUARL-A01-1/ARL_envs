from setuptools import setup, find_packages

setup(
    name="arl_envs",  # 你的包名
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # 在这里写依赖包，比如 "numpy", "mujoco", "gymnasium"
    ],
)