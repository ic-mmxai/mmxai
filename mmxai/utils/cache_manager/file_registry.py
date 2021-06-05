REGISTERED_FILE = []

EAST_TEXT_DETECTOR = {
    "download_name": "frozen_east_text_detection.tar.gz",
    "name": "frozen_east_text_detection.pb",
    "dir_path": "text_removal/text_detector/",
    "url": "https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1",
    "compression_method": "tar.gz"
}
REGISTERED_FILE.append(EAST_TEXT_DETECTOR)

def getResourceRecord(key: str):
    for entry in REGISTERED_FILE:
        if entry["name"] == key:
            return entry
    
    raise ValueError(f"Incorrect file key: {key} is not in mmxai cache registry!")