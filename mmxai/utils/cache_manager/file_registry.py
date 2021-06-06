REGISTERED_FILE = []

EAST_TEXT_DETECTOR = {
    "key": "EAST_detector",
    "download_name": "frozen_east_text_detection.tar.gz",
    "name": "frozen_east_text_detection.pb",
    "dir_path": "text_removal/text_detector/",
    "url": "https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1",
    "compression_method": "tar.gz",
    "from_google_drive": False,
    "google_drive_id": None,
}
REGISTERED_FILE.append(EAST_TEXT_DETECTOR)

FRCNN_CONFIG = {
    "key": "frcnn_config",
    "download_name": "frcnn.zip",
    "name": "config.yaml",
    "dir_path": "feature_selector/frcnn/",
    "url": "https://drive.google.com/u/0/uc?export=download&confirm=Ikrx&id=1iJ9D_sunUKiJaQWRb4iAY0LCUJu8wR3q",
    "compression_method": "zip",
    "from_google_drive": True,
    "google_drive_id": "1iJ9D_sunUKiJaQWRb4iAY0LCUJu8wR3q",
}
REGISTERED_FILE.append(FRCNN_CONFIG)

FRCNN_MODEL = {
    "key": "frcnn_model",
    "download_name": "frcnn.zip",
    "name": "model_finetuned.bin",
    "dir_path": "feature_selector/frcnn/",
    "url": "https://drive.google.com/u/0/uc?export=download&confirm=Ikrx&id=1iJ9D_sunUKiJaQWRb4iAY0LCUJu8wR3q",
    "compression_method": "zip",
    "from_google_drive": True,
    "google_drive_id": "1iJ9D_sunUKiJaQWRb4iAY0LCUJu8wR3q",
}
REGISTERED_FILE.append(FRCNN_MODEL)

def getResourceRecord(key: str, registry=REGISTERED_FILE):
    for entry in registry:
        if entry["key"] == key:
            return entry
    
    raise ValueError(f"Incorrect file key: {key} is not in mmxai cache registry!")