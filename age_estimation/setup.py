from setuptools import setup
setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='ageresnet',
    # Needed to actually package something
    packages=['ageresnet'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Ageresnet',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
