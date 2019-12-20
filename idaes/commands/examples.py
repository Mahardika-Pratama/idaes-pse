"""
Command to get IDAES examples
"""
# stdlib
from io import StringIO
import logging
from operator import itemgetter
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import List
from zipfile import ZipFile

# third-party
import click
import requests

# package
from idaes.commands.base import command_base
from idaes.ver import package_version as V

_log = logging.getLogger("idaes.commands.examples")


GITHUB = "https://github.com"
GITHUB_API = "https://api.github.com"
REPO_ORG = "idaes"
REPO_NAME = "examples-pse"
REPO_DIR = "src"
PKG_VERSION = f"{V.major}.{V.minor}.{V.micro}"
INSTALL_PKG = "idaes_examples"

# XXX: probably don't *really* want pre-releases, but they are
# XXX: handy for testing this script.
_skip_prereleases = False


class DownloadError(Exception):
    """Used for errors downloading the release files.
    """

    pass


class InstallError(Exception):
    """Used for errors installing the source as a Python package.
    """

    pass


@command_base.command(
    name="get-examples", help="Fetch example scripts and Jupyter Notebooks."
)
@click.option(
    "--dir", "-d", "directory", help="installation target directory", default="examples",
    type=str,
)
@click.option(
    "--no-install", "-I", "no_install", help="Do *not* install examples into 'idaes_examples' package",
    is_flag=True
)
@click.option(
    "--list-releases",
    "-l",
    help="List all available released versions, and stop",
    is_flag=True,
)
@click.option(
    "--no-download",
    "-N",
    "no_download",
    help="Do not download anything",
    is_flag=True
)
@click.option(
    "--version",
    "-V",
    help=f"Version of examples to download",
    default=PKG_VERSION,
    show_default=True,
)
def get_examples(directory, no_install, list_releases, no_download, version):
    """Get the examples from Github and put them in a local directory.
    """
    # list-releases mode
    if list_releases:
        releases = get_releases()
        print_releases(releases)
        sys.exit(0)
    # otherwise..
    target_dir = Path(directory)
    # no-download mode
    if no_download:
        _log.info("skipping download")
    else:
        click.echo("Downloading...")
        download_examples(get_releases(), target_dir, version)
        full_dir = os.path.realpath(target_dir)
        click.echo(f"Downloaded examples to directory '{full_dir}'")
    # install
    if not no_install:
        click.echo("Installing...")
        try:
            install_src(version, target_dir)
        except InstallError as err:
            click.echo(f"Install error: {err}")
            sys.exit(-1)
        click.echo(f"Installed examples in package {INSTALL_PKG}")


def download_examples(releases, target_dir, version):
    matched_release = None
    for rel in releases:
        if version == rel[1]:  # matches tag
            matched_release = rel
            break
    if matched_release is None:
        # modify message slightly depending on whether they selected a version
        if version == PKG_VERSION:
            how = "installed"
            how_ver = " and -V/--version to choose a desired version"
        else:
            how = "selected"
            how_ver = ""  # they already did this!
        click.echo(
            f"No release found matching {how} IDAES package version '{version}'."
        )
        click.echo(f"Use -l/--list-releases to see all{how_ver}.")
        sys.exit(-1)
    # check target directory
    if target_dir.exists():
        click.echo(f"Cannot download: target directory '{target_dir}' already exists.")
        sys.exit(-1)
    # download
    try:
        download_contents(version, target_dir)
    except DownloadError as err:
        click.echo(f"Download failed: {err}")
        shutil.rmtree(target_dir)  # remove partial download
        sys.exit(-1)


def download_contents(version, target_dir):
    """Download the given version from the Github releases and make
    its `REPO_DIR` subdirectory be the `target_dir`.

    Raises:
        DownloadError: if the GET on the release URL returns non-200 status
    """
    url = archive_file_url(version)
    _log.info(f"get examples from: {url}")
    # stream out to a big .zip file
    req = requests.get(url, stream=True)
    if req.status_code != 200:
        if req.status_code in (400, 404):
            raise DownloadError(f"file not found")
        raise DownloadError(f"status={req.status_code}")
    tmpdir = TemporaryDirectory()
    _log.debug(f"created temporary directory '{tmpdir.name}'")
    tmpfile = Path(tmpdir.name) / "examples.zip"
    with tmpfile.open("wb") as f:
        for chunk in req.iter_content(chunk_size=65536):
            f.write(chunk)
    _log.info(f"downloaded zipfile to {tmpfile}")
    # open as a zip file, and extract all files into the temporary directory
    _log.debug(f"open zip file: {tmpfile}")
    zipf = ZipFile(str(tmpfile))
    zipf.extractall(path=tmpdir.name)
    # move the REPO_DIR subdirectory into the target dir
    subdir = Path(tmpdir.name) / f"{REPO_NAME}-{version}" / REPO_DIR
    _log.debug(f"move {subdir} -> {target_dir}")
    os.rename(str(subdir), str(target_dir))
    _log.debug(f"removing temporary directory '{tmpdir.name}'")
    del tmpdir


def archive_file_url(version, org=REPO_ORG, repo=REPO_NAME):
    """Build & return URL for a given release version.
    """
    return f"{GITHUB}/{org}/{repo}/archive/{version}.zip"


def get_releases():
    """Returns a list of releases, with a tuple for each of (date, tag, info).
    The list is sorted in ascending order by date.
    """
    releases = []
    url = f"{GITHUB_API}/repos/{REPO_ORG}/{REPO_NAME}/releases"
    req = requests.get(url)
    for rel in req.json():
        if _skip_prereleases and rel["prerelease"]:
            continue
        releases.append((rel["published_at"], rel["tag_name"], rel["name"]))
    releases.sort(key=itemgetter(0))  # sort by publication date
    return releases


def print_releases(releases):
    """Print the releases, as returned by `get_releases()`, as a table
    to standard output.
    """
    if len(releases) == 0:
        print("No releases to list")
        return
    # determine column widths
    widths = [4, 7, 7]  # widths of column titles: date,version,details
    widths[0] = len(releases[0][0])  # dates are all the same
    # tags and names can have different widths
    for rel in releases:
        for i in range(1, 3):
            widths[i] = max(widths[i], len(rel[i]))
    # make row format
    pad = "  "
    fmt = f"{{date:{widths[0]}s}}{pad}{{tag:{widths[1]}s}}{pad}{{name:{widths[2]}s}}"
    # print header
    print("")
    print(fmt.format(date="Date", tag="Version", name="Details"))
    print(fmt.format(date="-" * widths[0], tag="-" * widths[1], name="-" * widths[2]))
    # print rows
    for rel in releases:
        print(fmt.format(date=rel[0], tag=rel[1], name=rel[2]))
    # print footer
    print("")


def install_src(version, target_dir):
    from setuptools import setup, find_packages
    root_dir = target_dir.parent
    examples_dir = root_dir / INSTALL_PKG
    if examples_dir.exists():
        raise InstallError(f"package directory {examples_dir} already exists")
    _log.info(f"install into {INSTALL_PKG} package")
    # set the args to make it look like the 'install' command has been invoked
    saved_args = sys.argv[:]
    sys.argv = ["setup.py", "install"]
    # add some empty __init__.py files
    _log.debug("add temporary __init__.py files")
    pydirs = find_python_directories(target_dir)
    pydirs.append(target_dir)  # include top-level dir
    for d in pydirs:
        init_py = d / "__init__.py"
        init_py.open("w")
    # temporarily rename target directory to the package name
    os.rename(target_dir, examples_dir)
    # if there is a 'build' directory, move it aside
    build_dir = root_dir / 'build'
    if build_dir.exists():
        from uuid import uuid1
        random_letters = str(uuid1())
        moved_build_dir = f"{build_dir}.{random_letters}"
        _log.debug(f"move existing build dir to {moved_build_dir}")
        os.rename(str(build_dir), moved_build_dir)
    else:
        _log.debug("no existing build directory (nothing to do)")
        moved_build_dir = None
    # run setuptools' setup command (in root directory)
    _log.info(f"run setup command in directory {root_dir}")
    orig_dir = os.curdir
    os.chdir(root_dir)
    packages = [d for d in find_packages() if d.startswith(INSTALL_PKG)]
    _log.debug(f"install packages: {packages}")
    # before running, grab stdout
    orig_stdout = sys.stdout
    sys.stdout = setup_out = StringIO()
    # run setup
    setup(
        name=INSTALL_PKG,
        version=version,
        # description='IDAES examples',
        packages=packages,
        python_requires=">=3.5,  <4",
        zip_safe=False
    )
    # restore stdout
    sys.stdout = orig_stdout
    # print/log output
    output_str = setup_out.getvalue()
    if _log.isEnabledFor(logging.DEBUG):
        for line in output_str.split("\n"):
            _log.debug(f"(setup) {line}")
    # name the target directory back to original
    os.rename(examples_dir, target_dir)
    # remove the empty __init__.py files
    # _log.debug("remove temporary __init__.py files")
    for d in pydirs:
        init_py = d / "__init__.py"
        init_py.unlink()
    # remove build dir, and restore any moved build dir
    shutil.rmtree(str(build_dir))
    if moved_build_dir is not None:
        _log.debug(f"restore build dir '{build_dir}' from '{moved_build_dir}'")
        os.rename(moved_build_dir, str(build_dir))
    # restore previous args
    sys.argv = saved_args
    # change back to previous directory
    os.chdir(orig_dir)


def find_python_directories(target_dir: Path) -> List[Path]:
    """Find all directories from target_dir, on down, that contain a
    Python module or sub-package.
    """
    # get directories that contain python files -> pydirs
    pydirs = set((x.parent for x in target_dir.rglob("*.py")))
    # get all directories in the tree leading to the 'pydirs'
    alldirs = set()
    for d in pydirs:
        while d != target_dir:
            alldirs.add(d)
            d = d.parent
    return list(alldirs)
