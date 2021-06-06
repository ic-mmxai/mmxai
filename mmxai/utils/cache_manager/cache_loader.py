import os
import requests
from tqdm import tqdm
import tarfile, zipfile

from mmxai.utils.cache_manager.file_registry import getResourceRecord, REGISTERED_FILE


def loadFromCache(key: str, registry=REGISTERED_FILE):
    """
    Using the key, locate the file from cache folder. If required file does not exit
    in the cache, will attempt to download and cache it.

    INPUTS:
        key - str: the file (record) key. The complete list of registered keys can be
            found in mmxai/utils/cache_manager/file_registry.py
        registry - list: list containing the record of registered files.
    
    RETURNS:
        str: absolute path to the cache file.

    """

    resource = getResourceRecord(key, registry=registry)

    cache_path = os.path.expanduser("~/.cache/mmxai")
    file_path = os.path.join(
        cache_path, resource["dir_path"], resource["name"])

    if not os.path.isfile(file_path):

        print(f"Cannot locate {os.path.join(resource['dir_path'], resource['name'])} " +
              f"in {cache_path}")
        download_path = os.path.join(
            cache_path, resource["dir_path"], resource["download_name"])

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        os.makedirs(os.path.dirname(download_path), exist_ok=True)

        downloadResource(resource, download_path)

        if resource["compression_method"]:
            extractFile(resource, download_path)

    assert os.path.isfile(
        file_path), f"ERROR: after downloading, {file_path} is still avaliable. Records in mmxai.utils.cache_manager.file_register.py might be wrong."

    return file_path


def downloadResource(resource, download_path):

    print(f"Downloading {download_path} from {resource['url']}")

    if resource["from_google_drive"]:
        downloadFromGoogleDrive(resource["google_drive_id"], download_path)
    else:
        downloadFromWeb(resource["url"], download_path)


def downloadFromWeb(url, download_path):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        block_size = 1024
        total_size = int(response.headers.get('content-length', 0))

        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(download_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=block_size):
                progress_bar.update(len(chunk))
                file.write(chunk)

        progress_bar.close()

        if total_size != 0 and progress_bar.n != total_size:
            os.remove(download_path)

            raise Exception(f"Downloaded file size different from the web content length. " +
                            f"Content length is {total_size}B. File size is {progress_bar.n}B. " +
                            "Try re-downloading.")


def downloadFromGoogleDrive(id, download_path):

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, download_path):
        chunk_size = 32768

        progress_bar = tqdm(total=None, unit='iB', unit_scale=True)

        with open(download_path, "wb") as file:
            for chunk in response.iter_content(chunk_size):
                if chunk: # filter out keep-alive new chunks
                    progress_bar.update(len(chunk))
                    file.write(chunk)
        
        progress_bar.close()

    google_drive_url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(google_drive_url, params = {'id': id}, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(google_drive_url, params = params, stream = True)

    save_response_content(response, download_path)



def extractFile(resource, download_path, remove=True):

    print(f"Extracting {download_path} ...")

    if resource["compression_method"] == "tar.gz":
        extrated_names = extractTarGz(download_path)
    elif resource["compression_method"] == "zip":
        extrated_names = extractZip(download_path)
    else:
        raise ValueError(
            f"Compression_method for {resource['name']} not supported. " +
            "Records in mmxai.utils.cache_manager.file_register.py might be wrong.")

    if remove:
        os.remove(download_path)

    print(f"... Extrated files are: {extrated_names}")


def extractTarGz(download_path):
    download_dir_path = os.path.dirname(download_path)

    with tarfile.open(download_path) as tar:
        extracted_names = tar.getnames()
        tar.extractall(path=download_dir_path)

    return extracted_names

def extractZip(download_path):
    download_dir_path = os.path.dirname(download_path)

    with zipfile.ZipFile(download_path, 'r') as zip:
        extracted_names = zip.namelist()
        zip.extractall(path=download_dir_path)
    
    return extracted_names