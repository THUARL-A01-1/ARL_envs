from setuptools import setup, find_packages

setup(
    name="arl-envs",  # 包名
    version="1.0",
    packages=find_packages(),
    install_requires=["opencv-python", 
                      "gymnasium", 
                      "matplotlib", 
                      "mujoco", 
                      "numpy==1.26.4", 
                      "open3d",  
                      "scipy"
                      ],
)
