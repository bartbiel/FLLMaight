from huggingface_hub import login, snapshot_download
from pathlib import Path
import grants

# Logowanie do Hugging Face przy użyciu tokena API
login(grants.API_HF)

# Tworzenie ścieżki do katalogu, gdzie zostaną zapisane pliki modelu
mistral_models_path = Path(grants.MistralDIR) / 'mistral_models' / '7B-Instruct-v0.3'
mistral_models_path.mkdir(parents=True, exist_ok=True)

# Pobieranie wybranych plików modelu z repozytorium Hugging Face
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    #allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"],
    allow_patterns=["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"],
    local_dir=str(mistral_models_path)  # można też podać Path bez konwersji, ale czasem lepiej na string
)
