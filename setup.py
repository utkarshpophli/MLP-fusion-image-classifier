from setuptools import find_packages, setup
from typing import List

HYERN_E_DOT = '-e .'
def requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYERN_E_DOT in requirements:
            requirements.remove(HYERN_E_DOT)
        
        return requirements

setup(
    name="mlp_fusion_image_classifier",
    version="0.0.1",
    author="Utkarsh",
    author_email="pophliutkarsh8@gmail.com",
    packages=find_packages(),
    install_requires=requirements("requirements.txt")
)