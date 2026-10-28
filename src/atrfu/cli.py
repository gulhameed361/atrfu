"""Command line: ``atrfu check`` and ``atrfu install-gjh``."""
import argparse
import platform
import sys

from . import __version__
from ._gjh import INSTALL_DIR, download_gjh, find_gjh


def check():
    """Report the versions and whether IPOPT and gjh are found; exit status 1
    if either is missing."""
    import numpy
    import pyomo
    from pyomo.environ import SolverFactory

    print(f"atrfu {__version__}, Python {platform.python_version()}, "
          f"Pyomo {pyomo.version.version}, NumPy {numpy.__version__}")
    ipopt = SolverFactory('ipopt')
    ok = ipopt.available(exception_flag=False)
    if ok:
        version = '.'.join(map(str, ipopt.version()))
        print(f"IPOPT : {ipopt.executable()} (version {version})")
    else:
        print("IPOPT : NOT FOUND -- conda install -c conda-forge ipopt "
              "(or idaes get-extensions)")
    gjh = find_gjh()
    print(f"gjh   : {gjh}" if gjh else
          "gjh   : not installed -- downloaded on the first solve, or now with "
          "'atrfu install-gjh'")
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog='atrfu', description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check', help='report versions and whether IPOPT and gjh are found')
    install = sub.add_parser('install-gjh', help="download AMPL's gjh for this platform")
    install.add_argument('--dir', default=INSTALL_DIR,
                         help=f'installation directory (default: {INSTALL_DIR})')
    args = parser.parse_args(argv)
    if args.command == 'check':
        return check()
    download_gjh(args.dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
