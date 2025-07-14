import joblib, os
import numpy as np
from services.logger import get_logger 


logger = get_logger("glove")
_glove_cache = None

def load_glove(path: str = "data/glove.6B.300d.txt"):
    global _glove_cache
    if _glove_cache is not None:
        return _glove_cache

    cache_path = path + ".pkl"
    if os.path.exists(cache_path):
        logger.info("⚡ [bold green]Loading cached GloVe vectors...[/]")
        _glove_cache = joblib.load(cache_path)
    else:
        logger.info("⏳ [yellow]Loading GloVe vectors from text...[/]")
        embeddings = {}
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vec = np.array(parts[1:], dtype=float)
                embeddings[word] = vec
        joblib.dump(embeddings, cache_path)
        _glove_cache = embeddings

    return _glove_cache

