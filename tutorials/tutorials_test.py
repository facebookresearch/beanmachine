from glob import glob

from bento.testutil import run_notebook
from libfb.py import testutil


class RunNotebookTest(testutil.BaseFacebookTestCase):
    def test_runs_without_errors(self):
        for notebook in glob("beanmachine/tutorials/*.ipynb"):
            run_notebook(notebook)
