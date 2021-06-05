import os
import requests
from tqdm import tqdm
import tarfile

from mmxai.utils.cache_manager.file_registry import getResourceRecord

def loadFromCache(name: str):
    
    resource = getResourceRecord(name)

    cache_path = os.path.expanduser("~/.cache/mmxai")
    file_path = os.path.join(cache_path, resource["dir_path"], resource["name"])

    if not os.path.isfile(file_path):

        print(f"Cannot locate {os.path.join(resource['dir_path'], resource['name'])} " +
        f"in {cache_path}")
        download_path = os.path.join(cache_path, resource["dir_path"], resource["download_name"])

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        os.makedirs(os.path.dirname(download_path), exist_ok=True)

        downloadResource(resource, download_path)

        if resource["compression_method"]:
            extractFile(resource, download_path, file_path)
        
    return file_path

def downloadResource(resource, download_path):
    
    print(f"Downloading {download_path} from {resource['url']}")
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

def extractFile(resource, download_path, file_path):

    print(f"Extracting {download_path} ...")

    if resource["compression_method"] == "tar.gz":
        extrated_path = extractTarGz(download_path)
    
    if extrated_path != file_path:
        os.replace(extrated_path, file_path)

    os.remove(download_path)

    print(f"... Extrated file saved to: {file_path}")

def extractTarGz(download_path):
    download_dir_path = os.path.dirname(download_path)

    with tarfile.open(download_path) as tar:
        extracted_name = tar.getnames()[0]
        tar.extractall(path=download_dir_path)
    
    extrated_path = os.path.join(download_dir_path, extracted_name)

    return extrated_path
