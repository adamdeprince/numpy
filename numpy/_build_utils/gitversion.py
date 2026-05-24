#!/usr/bin/env python3
import os
import textwrap


def init_version():
    init = os.path.join(os.path.dirname(__file__), '../../pyproject.toml')
    with open(init) as fid:
        data = fid.readlines()

    version_line = next(
        line for line in data if line.startswith('version =')
    )

    version = version_line.strip().split(' = ')[1]
    version = version.replace('"', '').replace("'", '')

    return version


def git_version(version):
    # Append last commit date and hash to dev version information,
    # if available

    import os.path
    import subprocess

    git_hash = ''
    git_date = ''
    try:
        p = subprocess.Popen(
            ['git', '-c', 'log.showSignature=false', 'log', '-1', '--format="%H %aI"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(__file__),
        )
    except FileNotFoundError:
        pass
    else:
        out, err = p.communicate()
        if p.returncode == 0:
            git_hash, git_date = (
                out.decode('utf-8')
                .strip()
                .replace('"', '')
                .split('T')[0]
                .replace('-', '')
                .split()
            )

    # Unofficial rebuilds (the dragon-array LoongArch/LASX wheels) mark
    # themselves with a PEP 440 local version label. The authoritative source
    # is the base version in pyproject.toml, e.g. "2.5.0.dev0+dragon.unofficial.1"
    # — that's the one place guaranteed to survive build front-ends (pypa/build
    # stages the source without .git, so a git/LOCAL_VERSION lookup misses).
    # If the base already carries a '+' segment, leave it: PEP 440 permits only
    # one, so we must not also append the git label.
    if '+' in version:
        return version, git_hash

    # Otherwise, a LOCAL_VERSION file at the repo root or the NUMPY_LOCAL_VERSION
    # env var can supply the label for in-tree builds; failing that, dev builds
    # get the usual git suffix.
    local = ''
    lv_file = os.path.join(os.path.dirname(__file__), '..', '..', 'LOCAL_VERSION')
    if os.path.isfile(lv_file):
        with open(lv_file) as f:
            local = f.read().strip()
    if not local:
        local = os.environ.get('NUMPY_LOCAL_VERSION', '').strip()

    if local:
        version += f'+{local}'
    elif git_hash and 'dev' in version:
        # Only attach git tag to development versions
        version += f'+git{git_date}.{git_hash[:7]}'

    return version, git_hash


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--write', help="Save version to this file")
    parser.add_argument(
        '--meson-dist',
        help='Output path is relative to MESON_DIST_ROOT',
        action='store_true'
    )
    args = parser.parse_args()

    version, git_hash = git_version(init_version())

    # For NumPy 2.0, this should only have one field: `version`
    template = textwrap.dedent(f'''
        """
        Module to expose more detailed version info for the installed `numpy`
        """
        version = "{version}"
        __version__ = version
        full_version = version

        git_revision = "{git_hash}"
        release = 'dev' not in version and '+' not in version
        short_version = version.split("+")[0]
    ''')

    if args.write:
        outfile = args.write
        if args.meson_dist:
            outfile = os.path.join(
                os.environ.get('MESON_DIST_ROOT', ''),
                outfile
            )

        # Print human readable output path
        relpath = os.path.relpath(outfile)
        if relpath.startswith('.'):
            relpath = outfile

        with open(outfile, 'w') as f:
            print(f'Saving version to {relpath}')
            f.write(template)
    else:
        print(version)
