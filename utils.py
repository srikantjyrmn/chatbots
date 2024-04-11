import requests

def download_file(url, local_filename):
    """
    Downloads a file from the specified URL and saves it to the local filesystem.

    Parameters:
    - url: The URL of the file to download.
    - local_filename: The local path where the file should be saved.

    Returns:
    None
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
                