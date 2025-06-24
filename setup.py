#This is a setup script for the Vehicle Annotation Toolbox project.
#It uses setuptools to package the application and specify its dependencies.

from setuptools import setup, find_packages

setup(
    name="vehicle-annotation-toolbox",
    version="1.0.0",
    description="Vehicle annotation toolbox with YOLO11 OBB detection and BoTSORT tracking",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "PySide6>=6.5.0",
        "opencv-python>=4.8.0"
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "ultralytics>=8.0.0",
        "supervision>=0.16.0"
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0"
    ],
    entry_points={
        "console_scripts": [
            "vat=main:main",
        ],
    },
)