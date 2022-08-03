from setuptools import setup, find_namespace_packages

with open("README.md", "r") as f:
  long_description = f.read()

setup(name="NeurIPS22-CellSeg",
			packages=find_namespace_packages(include=["baseline", "baseline.*","models"]),
      version="0.0.1",
      description="NeurIPS-CellSeg-baseline",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/JunMa11/NeurIPS-CellSeg",
      author="Cheng Ge",
      author_email="13851520957@163.com",
      license="MIT",   
      platforms=["all"],
      install_requires=[
                        "monai",
						"numpy",
						"nibabel",
						"scikit-image",
						"pillow",
						"tensorboard",
						"gdown",
						"torchvision",
						"tqdm",
						"psutil",
						"pandas",
						"einops",
      ],
      entry_points={
          'console_scripts': [
          		'pre_process_3class = data.pre_process_3class:main',
          		'model_training_3class = baseline.model_training_3class:main',
          		'predict = baseline.predict:main',
          ],
      },
      )