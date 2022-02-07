import urllib.request
from tqdm import tqdm
import os
import zipfile

### Download the dataset

def download_hook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.request.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to

def download(url, save_dir):
    filename = url.split('/')[-1]
    with tqdm(unit = 'B', unit_scale = True, unit_divisor = 1024, miniters = 1, desc = filename) as t:
        urllib.request.urlretrieve(url, filename = os.path.join(save_dir, filename), reporthook = download_hook(t), data = None)

if __name__ == '__main__':
    # Download ground truth
    if not os.path.exists("CMP_facade_DB_base.zip"):
        download("http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip", "./")
    # Extract in the correct folder
    with zipfile.ZipFile("CMP_facade_DB_base.zip", 'r') as zip_ref:
        zip_ref.extractall("./facades")
        os.rename("./facades/base", "./facades/train")

    # Download ground truth
    if not os.path.exists("CMP_facade_DB_extended.zip"):
        download("http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip", "./")
    # Extract in the correct folder
    with zipfile.ZipFile("CMP_facade_DB_extended.zip", 'r') as zip_ref:
        zip_ref.extractall("./facades")
        os.rename("./facades/extended", "./facades/val")
