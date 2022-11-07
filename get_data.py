import os
import requests
import pandas as pd

from concurrent.futures import ThreadPoolExecutor


def download_photo(url, to):
    to = f"data/unsplash/{url.split('/')[-1]}.jpg"
    data = requests.get(url)
    
    with open(to, 'wb') as f:
        f.write(data)
    f.close()


def download_unsplash_dataset(max_workers=os.cpu_count()):
    urls = pd.read_csv('data/csvs/photos.tsv')['photo_url']

    if not os.path.exists('data/unsplash'):
        os.makedirs('data/unsplash')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(download_photo, urls)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--dataset', required=True)

    args = parser.parse_args()

    if args.dataset == 'unsplash':
        assert os.path.exists('data/csvs/photos/tsv'), 'photos tsv does not exist'

        download_unsplash_dataset()