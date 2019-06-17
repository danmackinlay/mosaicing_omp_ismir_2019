from setuptools import find_packages, setup

long_description = """
pattern_machine
"""

config = dict(
    description='machine listening',
    author='Anonymous',
    url='URL to get it at.',
    download_url='Where to download it.',
    author_email='My email.',
    version='0.1',
    install_requires=[''],
    packages=find_packages(exclude=('tests', )),
    scripts=[],
    name='pattern_machine',
)

setup(**config)
