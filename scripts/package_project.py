"""Package portable deliverables without duplicating large raw daily caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root=Path(__file__).resolve().parents[1]
out=root.parent/'HelsinkiTraffic_Project.zip'
with ZipFile(out,'w',ZIP_DEFLATED,compresslevel=6) as archive:
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(root)
        if any(part in {'__pycache__','.venv','.ipynb_checkpoints'} for part in rel.parts):continue
        if rel.parts[:2] in [('data','raw'),('data','daily')]:continue
        archive.write(path,Path(root.name)/rel)
print(out, out.stat().st_size)
with ZipFile(out) as archive:
    assert archive.testzip() is None
    print('Archive integrity passed;',len(archive.namelist()),'files')
