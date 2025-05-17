from setuptools import setup, find_packages

setup(
    name='HolyPipette-PBL',
    version='0.1',
    description='Deep Learning guided Automated Patch Clamp Electrophysiology System',
    url='https://github.com/romainbrette/holypipette/',
    author='Benjamin Magondu, Nathan Malta, Kaden StillWagon, Victor Guyard',
    author_email='bmagondu3@gatech.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(),
    install_requires=['numpy', 'PyQt5', 'qtawesome', 'pillow', 'pyserial',
                      'param', 'pyyaml']
)