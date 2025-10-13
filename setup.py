import setuptools

with open('README.md', mode='r', encoding='utf-8', errors='ignore') as fh:
    long_description = fh.read()
    

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setuptools.setup(
    name='pccdr',
    version='1.0.3',
    author='Jacob Gildenblat',
    author_email='jacob.gildenblat@gmail.com',
    description='Dimensionality reduction by preserving clusters and correlations.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jacobgil/pcc',
    project_urls={
        'Bug Tracker': 'https://github.com/jacobgil/pcc/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    packages=setuptools.find_packages(
            include=["pcc"]),
    install_requires=requirements)
