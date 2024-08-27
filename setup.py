from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quickprompt",
    version="0.1.0",
    author="Abhijit Das",
    author_email="aj.das.research@gmail.com",
    description="A tool for quickly generating high-quality prompts for various AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quickprompt",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "openai>=0.27.0",
        "anthropic>=0.3.0",
        "groq>=0.3.4",
    ],
    entry_points={
        "console_scripts": [
            "quickprompt=quickprompt.cli:main",
        ],
    },
)