import setuptools

with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="summarization_annotation_tool",
    version="0.1",
    package_data={"summarization_annotation": ["resources/*.js", "resources/*.css"]},
    include_package_data=True,
    description="Annotation tool for summarization project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.soma.salesforce.com/summarization/annotation",
    packages=setuptools.find_packages(),
    install_requires=required,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "summarization-annotation=summarization_annotation.runner:main"
        ]
    },
)
