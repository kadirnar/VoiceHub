import os
import platform

def get_cache_dir():
    cache_dir = os.path.join(
        os.getenv('APPDATA') if platform.system() == "Windows" 
        else os.path.expanduser("~/.cache"), 
        "outeai"
    )
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir