from mmxai.utils.cache_manager.file_registry import REGISTERED_FILE, getResourceRecord

MOCK_REGISTRY = [
    {
        "key": "test_archive_web",
        "download_name": "test_archive.tar.gz",
        "name": "test_archive.tar.gz",
        "dir_path": "tests",
        "url": "https://github.com/ic-mmxai/mmxai/raw/cache-management/tests/mmxai/utils/cache_manager/test_archive.tar.gz",
        "compression_method": "",
        "from_google_drive": False,
        "google_drive_id": "",
    },
    {
        "key": "test_archive_local",
        "download_name": "test_archive.tar.gz",
        "name": "test_archive.tar.gz",
        "dir_path": "tests",
        "url": "",
        "compression_method": "",
        "from_google_drive": False,
        "google_drive_id": "",
    },
    {
        "key": "test_txt_google_drive",
        "download_name": "test_file_to_download.txt",
        "name": "test_file_to_download.txt",
        "dir_path": "tests",
        "url": "https://drive.google.com/file/d/1NL73ptClPDuOHUCHGIzL1FqfIay2Kthn/view?usp=sharing",
        "compression_method": "",
        "from_google_drive": True,
        "google_drive_id": "1NL73ptClPDuOHUCHGIzL1FqfIay2Kthn",
    }
]

def testRegisterRecordsHaveCorrectEntries():
    for record in REGISTERED_FILE:
        keys = record.keys()

        assert "key" in keys
        assert "download_name" in keys
        assert "name" in keys
        assert "dir_path" in keys
        assert "url" in keys
        assert "compression_method" in keys
        assert "from_google_drive" in keys
        assert "google_drive_id" in keys

def testGetResourceRecord():
    record = getResourceRecord("test_archive_web", registry=MOCK_REGISTRY)

    assert record == MOCK_REGISTRY[0]

    try:
        getResourceRecord("wrong_key", registry=MOCK_REGISTRY)
    except ValueError:
        pass
    else:
        assert False, "ERROR: using wrong key should raise ValueError"

if __name__ == "__main__":
    testRegisterRecordsHaveCorrectEntries()
    testGetResourceRecord()