from setuptools import setup, find_packages

setup(
    name="zed_kb",
    version="0.1.0",
    packages=find_packages(),
    description="Secure AI powered internal knowledge base with tiered access and authorization controls",
    author="ZedKB Team",
    python_requires=">=3.8",
    install_requires=[
        # Dependencies are defined in requirements.txt
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Enterprise",
        "Topic :: Knowledge Management :: Document Processing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)