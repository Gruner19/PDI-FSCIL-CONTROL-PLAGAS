import numpy as np
import json
from typing import Dict, List, Optional


class NumpyEncoder(json.JSONEncoder):
    """Encoder JSON para tipos numpy."""
    def default(self, objeto):
        if isinstance(objeto, (np.integer,)):
            return int(objeto)
        if isinstance(objeto, (np.floating,)):
            return float(objeto)
        if isinstance(objeto, np.ndarray):
            return objeto.tolist()
        return super().default(objeto)


def dicionario_para_json(dicionario: Dict) -> str:
    """Serializa dicionário com valores numpy para JSON."""
    return json.dumps(dicionario, cls=NumpyEncoder, indent=2, ensure_ascii=False)
