from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MTG_DIR = DATA_DIR / "mtg_jamendo"
MTG_METADATA_REPO = MTG_DIR / "metadata_repo"

TRAINING_AUDIO_DIR = DATA_DIR / "training_audio"
TRAINING_PROCESSED_DIR = DATA_DIR / "training_processed"

SETLIST_AUDIO_DIR = DATA_DIR / "setlist_audio"
SETLIST_PROCESSED_DIR = DATA_DIR / "setlist_processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREVIEW_DIR = OUTPUTS_DIR / "previews"

MODELS_DIR = PROJECT_ROOT / "models"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aiff", ".aif"}