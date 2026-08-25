#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
COMMAND=${1:-install}

say() { printf '%s\n' "[10ary-One-Bit] $*"; }
die() { printf '%s\n' "[10ary-One-Bit] ERROR: $*" >&2; exit 1; }
need_python() { command -v python3 >/dev/null 2>&1 || die "python3 is required"; python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || die "Python 3.11 or newer is required"; }
make_venv() { need_python; if [ ! -x "$PY" ]; then say "creating isolated environment"; python3 -m venv "$VENV" || die "python3-venv is missing; install it with your OS package manager"; fi; }
install_deps() { make_venv; say "installing/repairing pinned dependencies"; "$PY" -m pip install --disable-pip-version-check --upgrade 'pip==25.2' 'setuptools==80.9.0' 'wheel==0.45.1'; "$PIP" install --disable-pip-version-check -r "$ROOT/requirements.txt"; }
verify() { "$PY" -c 'import numpy, PySide6; print("dependencies: OK")'; (cd "$ROOT" && "$PY" -m compileall -q tenary tests && "$PY" -m unittest discover -s tests -v && "$PY" -m tenary self-test && "$PY" -m tenary.gui --demo --repeats 5); }
doctor() { need_python; [ -x "$PY" ] || die "environment missing; run ./install.sh install"; verify; say "doctor: PASS"; }
backup() { out=${2:-"$ROOT/10ary-one-bit-backup-$(date +%Y%m%dT%H%M%S).tar.gz"}; find "$ROOT" -maxdepth 1 -type f -name '*.t10b' -print | tar -czf "$out" -T -; say "backup=$out"; }
restore() { archive=${2:-}; [ -n "$archive" ] && [ -f "$archive" ] || die "usage: ./install.sh restore BACKUP.tar.gz"; tar -tzf "$archive" | awk '/(^|\/)\.\.($|\/)|^\// {bad=1} END {exit bad}' || die "unsafe backup paths"; tar -xzf "$archive" -C "$ROOT"; say "restore complete"; }

case "$COMMAND" in
  install|repair) install_deps; verify ;;
  start|restart) make_venv; verify ;;
  stop) say "no background service is used; nothing to stop" ;;
  status) if [ -x "$PY" ]; then "$PY" --version; say "environment: ready"; else say "environment: not installed"; exit 1; fi ;;
  migrate) install_deps; say "T10B1 v1 artifacts require no migration" ;;
  backup) backup "$@" ;;
  restore) restore "$@" ;;
  logs) find "$ROOT" -maxdepth 1 -type f -name '*.t10b' -print ;;
  doctor) doctor ;;
  uninstall) rm -rf "$VENV"; say "environment removed; model artifacts preserved" ;;
  test) make_venv; verify ;;
  *) die "usage: ./install.sh {install|start|stop|restart|status|repair|migrate|backup [FILE]|restore FILE|logs|doctor|uninstall|test}" ;;
esac
