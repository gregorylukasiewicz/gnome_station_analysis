from distutils.core import setup
from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
name="gnome_station_analysis",
    version="0.1.4",
    author="GNOME team in Kraków",
    author_email="gregory.lukasiewicz@gmail.com",
    packages=["gnome_station_analysis"],
    #scripts = glob('bin/*'),
    url="https://github.com/gregorylukasiewicz/gnome_station_analysis",
    download_url = "https://github.com/gregorylukasiewicz/gnome_station_analysis/archive/refs/tags/v_0.1.3.tar.gz",
    description="GNOME Station Analysis Tools",
    install_requires=["numpy", "matplotlib", "regex", "glob2", "scipy"],
    long_description = long_description,
    long_description_content_type = "text/markdown"
)
