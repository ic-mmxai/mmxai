from mmxai.utils.cache_manager.cache_loader import loadFromCache, extractFile

from test_file_registry import MOCK_REGISTRY

import os
import shutil


def testCanExtractTarGz():

    resource = {"compression_method": "tar.gz"}

    extractFile(
        resource, "tests/mmxai/utils/cache_manager/test_archive.tar.gz", remove=False)

    checkExtratedTestFilesExist()


def testCanExtractZip():

    resource = {"compression_method": "zip"}

    extractFile(
        resource, "tests/mmxai/utils/cache_manager/test_archive.zip", remove=False)

    checkExtratedTestFilesExist()


def testRaiseErrorForUnsupportedExtrat():
    resource = {"name": "wrong_file", "compression_method": "wrong_method"}

    try:
        extractFile(
            resource, "tests/mmxai/utils/cache_manager/test_file_to_download.txt", remove=False)
    except ValueError:
        pass
    else:
        assert False, "ERROR: wrong compression methods should raise ValueError."


def checkExtratedTestFilesExist():
    assert os.path.isfile(
        "tests/mmxai/utils/cache_manager/test_archive/file1.txt")
    assert os.path.isfile(
        "tests/mmxai/utils/cache_manager/test_archive/file2.txt")

    shutil.rmtree("tests/mmxai/utils/cache_manager/test_archive")


def checkCanDownloadFromWebAndReuseFromCache():

    path = loadFromCache("test_archive_web", registry=MOCK_REGISTRY)
    assert path == os.path.expanduser(
        "~/.cache/mmxai/tests/test_archive.tar.gz")

    path = loadFromCache("test_archive_local", registry=MOCK_REGISTRY)
    assert path == os.path.expanduser(
        "~/.cache/mmxai/tests/test_archive.tar.gz")

    shutil.rmtree(os.path.expanduser("~/.cache/mmxai/tests/"))


def checkCanDownloadFromGoogleDrive():
    path = loadFromCache("test_txt_google_drive", registry=MOCK_REGISTRY)
    assert path == os.path.expanduser(
        "~/.cache/mmxai/tests/test_file_to_download.txt")


if __name__ == "__main__":
    testCanExtractTarGz()
    testCanExtractZip()
    testRaiseErrorForUnsupportedExtrat()

    checkCanDownloadFromWebAndReuseFromCache()
    checkCanDownloadFromGoogleDrive()
