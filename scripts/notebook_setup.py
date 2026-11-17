"""Self-contained setup copied into the notebook's first executable cell."""
from pathlib import Path
import importlib.metadata
import os
import subprocess
import sys
import tempfile

REPOSITORY_URL = 'https://github.com/vipin-143/Traffic-Congestion-Prediction.git'

def _is_project(path):
    return all((path / name).is_file() for name in (
        'traffic/__init__.py', 'data/traffic_weather.csv.gz', 'artifacts/models.joblib'))

def _find_project(start):
    # Search the current directory, its ancestors and their immediate children.
    # No recursive scan of Drive or the entire computer is needed.
    for parent in [start, *start.parents]:
        if _is_project(parent):
            return parent
        try:
            for child in sorted(parent.iterdir()):
                if child.is_dir() and _is_project(child):
                    return child
        except (PermissionError, OSError):
            continue
        if parent == parent.parent:
            break
    return None

PROJECT_ROOT = _find_project(Path.cwd().resolve())
if PROJECT_ROOT is None:
    # A downloaded ipynb does not contain the traffic package, data or models.
    # Fetch the complete public repository automatically without credentials.
    destination = Path.cwd() / 'Traffic-Congestion-Prediction'
    if destination.exists():
        destination = Path(tempfile.mkdtemp(prefix='helsinki-traffic-')) / 'project'
    print('Downloading the complete project from GitHub...')
    try:
        subprocess.run(['git', 'clone', '--depth', '1', REPOSITORY_URL, str(destination)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('Could not download the project. Check internet access and Git, or download and extract the complete repository ZIP, then run this notebook inside it.') from exc
    if not _is_project(destination):
        raise RuntimeError('Downloaded repository is missing required project files. Check the repository URL and branch.')
    PROJECT_ROOT = destination.resolve()

os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IN_COLAB = 'google.colab' in sys.modules or bool(os.environ.get('COLAB_RELEASE_TAG'))
if IN_COLAB:
    # Match the environment that created the serialized scikit-learn pipeline.
    required = ['numpy', 'pandas', 'scikit-learn', 'joblib', 'requests', 'matplotlib']
    pins = dict(line.strip().split('==', 1) for line in (PROJECT_ROOT / 'requirements-lock.txt').read_text().splitlines() if '==' in line)
    install = []
    for package in required:
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != pins[package]:
            install.append(f'{package}=={pins[package]}')
    if install:
        print('Installing the notebook dependencies. If Colab requests a runtime restart, restart and run all cells again.')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *install], check=True)

print('Project ready:', PROJECT_ROOT)
print('Run the remaining cells in order. No API key is required.')
