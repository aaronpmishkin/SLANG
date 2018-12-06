from setuptools import setup, find_packages

setup(
    name='slang_torchutils', 
    version='0.1.0',
    description='Some extensions/helpers for common functionality of PyTorch',  
    long_description="", 
    long_description_content_type='text/markdown',
    url='https://github.com/aaronpmishkin/SLANG',
    author='Frederik Kunstner, Aaron Mishkin, Didrik Nielsen',
    author_email='frederik.kunstner@gmail.com, amishkin@cs.ubc.ca, didrik.nielsen@riken.jp',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
		'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='torch pytorch torchutils',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'other']),
    install_requires=['torch'],
)
