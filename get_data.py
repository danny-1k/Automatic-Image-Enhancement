import os
import requests
import pandas as pd

from concurrent.futures import ThreadPoolExecutor


def download_photo(url, dataset):
    to = f"data/{dataset}/{url.split('/')[-1]}.jpg"

    if not os.path.exists(to):

        try:

            r = requests.get(url, allow_redirects=True)
            #h = r.headers
            data = r.content

            with open(to, 'wb') as f:
                f.write(data)
            f.close()

        except Exception as e:
            print('error occured ...', e)


def download_unsplash_dataset(max_workers):
    urls = pd.read_csv('data/csvs/photos.tsv', delimiter='\t')['photo_image_url']
    print('read in.')
    if not os.path.exists('data/unsplash'):
        os.makedirs('data/unsplash')


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(download_photo, urls, ['unsplash']*len(urls))



if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num_workers', required=False, default=os.cpu_count(), type=int)

    args = parser.parse_args()
    num_workers = args.num_workers

    if args.dataset == 'unsplash':
        assert os.path.exists('data/csvs/photos.tsv'), 'photos tsv does not exist'

        download_unsplash_dataset(num_workers)
