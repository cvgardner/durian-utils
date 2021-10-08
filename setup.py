from setuptools import setup

setup(name='durian_utils',
      version='0.1',
      description='tools and wrappers used to create durian data projects',
      url='https://github.com/cvgardner/durian_utils',
      author='DurianIRL',
      author_email='therealdurianirl@gmail.com',
      license='MIT',
      packages=['durian_utils'],
      install_requires=[
          'streamlink',
          'pandas',
          'numpy',
          'imutils',
          'opencv-python',
          'pytesseract',
          'tqdm',
          'fuzzywuzzy'
      ],
      zip_safe=False)