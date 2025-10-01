from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agsac",
    version="0.1.0",
    author="AGSAC Team",
    author_email="agsac@example.com",
    description="基于注意力机制和几何微分评估的机器狗导航系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agsac/agsac_dog_navigation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "gazebo": [
            "gazebo-ros>=3.0.0",
            "rospy>=1.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agsac-train=scripts.train:main",
            "agsac-eval=scripts.evaluate:main",
            "agsac-infer=scripts.inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "agsac": ["configs/*.yaml"],
    },
)
