import importlib.util
import os


def test_can_import_show_output():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root, "scripts", "show_output.py")
    spec = importlib.util.spec_from_file_location("show_output", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # should not raise
