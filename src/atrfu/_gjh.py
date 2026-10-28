"""AMPL's gjh: objective gradient and constraint Jacobian of a Pyomo model.

Registers the Pyomo pseudo-solver 'local.gjh' and finds or installs the gjh
executable. gjh is looked up, in order, at

1. the path in the ``ATRFU_GJH`` environment variable;
2. the ``ampl_module_gjh`` package, if installed
   (``pip install ampl-module-gjh --extra-index-url https://pypi.ampl.com``);
3. the system PATH;
4. ``<Pyomo config dir>/bin``, where a download is installed.

If none is found, ``ensure_gjh`` downloads AMPL's ``ampl-module-gjh`` wheel
for this platform from AMPL's package index and installs the executable in
the Pyomo config dir.
"""
import io
import logging
import os
import re
import shutil
import stat
import sys
import zipfile
from urllib.request import Request, urlopen

from pyomo.common.envvar import PYOMO_CONFIG_DIR
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.base import SolverFactory
from pyomo.solvers.plugins.solvers.ASL import ASL

from ._readgjh import readgjh

logger = logging.getLogger('atrfu.gjh')

_WINDOWS = sys.platform.startswith(('win', 'cygwin'))
GJH_FILENAME = 'gjh.exe' if _WINDOWS else 'gjh'
INSTALL_DIR = os.path.join(PYOMO_CONFIG_DIR, 'bin')

_GJH_INDEX = 'https://pypi.ampl.com/ampl-module-gjh'
# Used when the index page cannot be parsed
_GJH_FALLBACK_VERSION = '20241202'
# AMPL's index rejects the default urllib User-Agent
_HTTP_HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; atrfu)'}


def _is_valid_exe(path):
    """A non-empty file (an interrupted download can leave an empty one)."""
    return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 0


def _candidates():
    yield os.environ.get('ATRFU_GJH')
    try:
        import ampl_module_gjh
        yield os.path.join(ampl_module_gjh.bin_dir, GJH_FILENAME)
    except ImportError:
        pass
    yield shutil.which('gjh')
    yield os.path.join(INSTALL_DIR, GJH_FILENAME)


def find_gjh():
    """Absolute path of the first usable gjh executable, or None."""
    return next((os.path.abspath(p) for p in _candidates() if _is_valid_exe(p)),
                None)


def _platform_tag():
    import platform
    if _WINDOWS:
        return 'win_amd64'
    if sys.platform.startswith('darwin'):
        return 'macosx_10_9_universal2'
    if platform.machine().lower() in ('aarch64', 'arm64'):
        return 'manylinux2014_aarch64'
    return 'manylinux2010_x86_64'


def _http_get(url, timeout=120):
    with urlopen(Request(url, headers=_HTTP_HEADERS), timeout=timeout) as r:
        return r.read()


def _wheel_url():
    """URL of the newest ampl-module-gjh wheel for this platform."""
    tag = _platform_tag()
    try:
        page = _http_get(_GJH_INDEX + '/', timeout=60).decode('utf-8', 'replace')
        wheels = sorted(set(re.findall(
            rf"""href=['"]([^'"]*ampl_module_gjh-[^'"]*{re.escape(tag)}\.whl)""", page)))
        if wheels:    # names embed YYYYMMDD versions: the last is the newest
            return f"{_GJH_INDEX}/{wheels[-1].rsplit('/', 1)[-1]}"
    except Exception as e:
        logger.warning("gjh index lookup failed (%s); using version %s",
                       e, _GJH_FALLBACK_VERSION)
    return f'{_GJH_INDEX}/ampl_module_gjh-{_GJH_FALLBACK_VERSION}-py3-none-{tag}.whl'


def download_gjh(dest_dir=INSTALL_DIR):
    """Download gjh for this platform into ``dest_dir`` and return its path.

    The executable is written to a temporary name and moved into place only
    when complete, so a failed download never leaves a broken gjh behind.
    """
    url = _wheel_url()
    print(f"[gjh] downloading {url}")
    with zipfile.ZipFile(io.BytesIO(_http_get(url))) as zf:
        member = next((n for n in zf.namelist()
                       if n.endswith('/bin/' + GJH_FILENAME)), None)
        if member is None:
            raise RuntimeError(f"no {GJH_FILENAME} in {url}")
        exe_bytes = zf.read(member)

    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, GJH_FILENAME)
    tmp = dest + '.download'
    try:
        with open(tmp, 'wb') as f:
            f.write(exe_bytes)
        os.chmod(tmp, os.stat(tmp).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"[gjh] installed {dest}")
    return dest


def ensure_gjh(download_if_missing=True, verbose=False):
    """Return the path of a usable gjh, downloading it if needed.

    The executable's directory is put on PATH for this process, which is
    where Pyomo's ASL interface looks for it.

    Parameters
    ----------
    download_if_missing : bool
        Download gjh when none is found (otherwise raise RuntimeError).
    verbose : bool
        Print the path in use.
    """
    exe = find_gjh()
    if exe is None:
        if not download_if_missing:
            raise RuntimeError(
                "gjh not found (ATRFU_GJH, ampl_module_gjh, PATH, "
                f"{INSTALL_DIR}). Run 'atrfu install-gjh', or "
                "'pip install ampl-module-gjh --extra-index-url https://pypi.ampl.com'.")
        try:
            exe = download_gjh()
        except Exception as e:
            raise RuntimeError(
                f"gjh is not installed and could not be downloaded ({e}). Install it "
                "with 'pip install ampl-module-gjh --extra-index-url "
                "https://pypi.ampl.com', or set ATRFU_GJH to a gjh executable.") from e

    exe_dir = os.path.dirname(exe)
    if exe_dir not in os.environ.get('PATH', '').split(os.pathsep):
        os.environ['PATH'] = exe_dir + os.pathsep + os.environ.get('PATH', '')
    if verbose:
        print(f"[gjh] using {exe}")
    return exe


@SolverFactory.register('local.gjh', doc='AMPL gjh: gradient and Jacobian evaluation')
class LocalGJHSolver(ASL):
    def __init__(self, **kwds):
        kwds['type'] = 'gjh'
        kwds['symbolic_solver_labels'] = True
        super().__init__(**kwds)
        self.options.solver = 'gjh'
        self._metasolver = False

    def available(self, exception_flag=True):
        exe = find_gjh()
        if exe:
            self.options.solver = exe
            return super().available(exception_flag=exception_flag)
        if exception_flag:
            raise RuntimeError("gjh not found; run atrfu.ensure_gjh() to install it.")
        return False

    def _initialize_callbacks(self, model):
        self._model = model
        self._model._gjh_info = None
        super()._initialize_callbacks(model)

    def _presolve(self, *args, **kwds):
        super()._presolve(*args, **kwds)
        self._gjh_file = self._soln_file[:-3] + 'gjh'
        TempfileManager.add_tempfile(self._gjh_file, exists=False)

    def _postsolve(self):
        stem = self._gjh_file[:-3]
        if not all(os.path.exists(stem + ext) for ext in ('gjh', 'col', 'row')):
            raise RuntimeError("gjh did not write its .gjh, .col and .row files.")
        self._model._gjh_info = readgjh(self._gjh_file)
        self._model = None
        return super()._postsolve()
