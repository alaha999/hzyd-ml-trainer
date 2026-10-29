"""Load a model implementation given its local .py path."""
import importlib.util
from pathlib import Path

def load_model_module(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix != '.py':
        raise FileNotFoundError(f'Invalid model module: {path}')
    spec = importlib.util.spec_from_file_location('hep_selected_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('build_model', 'predict'):
        if not hasattr(module, name):
            raise AttributeError(f'{path} must define {name}()')
    return module
