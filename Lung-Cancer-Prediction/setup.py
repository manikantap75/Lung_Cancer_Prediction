from setuptools import setup, find_packages

setup(
    name="lung-cancer-prediction",
    version="1.0.0",
    description="Machine Learning Clinical Decision Support Framework for Early Lung Cancer Risk Stratification",
    author="Clinical ML Research Group",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
        "streamlit>=1.30.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
