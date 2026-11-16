"""Exercise all local dashboard pages with Streamlit's real application runner."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

root=Path(__file__).resolve().parents[1]
app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
assert not app.exception, str(app.exception)
for page in ['Live forecast','Model evaluation','Notebook','Data and sources','Historical demo']:
    app.sidebar.radio[1].set_value(page).run()
    assert not app.exception, f'{page}: {app.exception}'
    print('PASS',page,flush=True)
print('All local dashboard pages passed.')
