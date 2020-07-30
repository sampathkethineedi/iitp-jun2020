from setuptools import setup
setup(
    name = 'Paraphrase-CLI',
    version = '0.1.0',
    packages = ['Paraphrase'],
    entry_points = {
        'console_scripts': [
            'Paraphrase = Paraphrase.__main__:main'
        ]
    })