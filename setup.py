import setuptools
from setuptools import find_packages,setup
from typing import List

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()



HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


__version__ = "0.0.0"

REPO_NAME = "brain_tumor_Unet"
AUTHOR_USER_NAME = "Vampaxx"
SRC_REPO = "brain_tumor"
AUTHOR_EMAIL = "arjunappu1001@gmail.com"


setuptools.setup(
    name                    = SRC_REPO,
    version                 = __version__,
    author                  = AUTHOR_USER_NAME,
    author_email            = AUTHOR_EMAIL,
    description             = "brain tumor masking project image segmentation",
    long_description        = long_description,
    long_description_content = "text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Brain tumor" : f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'))