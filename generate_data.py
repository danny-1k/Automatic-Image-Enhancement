import os
from models import ImageSelector
from data_utils import read_img, random_manipulation
import pandas as pd

from concurrent.futures import ThreadPoolExecutor


class DataGenerator:
    def __init__(self, images_path, to_path, var_per_image, num_threads, max_samples):
        self.images_path = images_path
        self.to_path = to_path
        self.images_path = images_path
        self.max_samples = max_samples
        self.num_threads = num_threads
        self.num_sampled = 0


        self.var_per_image = var_per_image

        self.image_selector = ImageSelector()


        self.df = pd.DataFrame({}, columns=['img_id', 'brightness', 'saturation', 'contrast', 'sharpness'], index=['img_id'])


    def process_image(self, f):
        img = read_img(f)

        for _ in  range(self.var_per_image):

            new_img, params = self.get_random_image(img)

            new_img.save(f'{self.to_path}/images/{self.num_sampled}.jpg')

            self.df = self.df.append({
                'img_id': self.num_sampled,
                'brightness': params[0],
                'saturation': params[1],
                'contrast': params[2],
                'sharpness': params[3]
            })

            self.num_sampled += 1

            if self.num_sampled >= self.max_samples:
                break


    def get_random_image(self, img):
        new_img, params = random_manipulation(img)

        if self.image_selector(new_img)[0]:
            return new_img, params

        while True:
            new_img, params = random_manipulation(img)

            if self.image_selector(new_img)[0]:

                return new_img, params

    
    def run(self):
        if not os.path.exists(self.to_path):
            os.makedirs(self.to_path)
        else:
            assert os.listdir(self.to_path) == 0, f'There are already Images in {self.to_path}'

        img_paths = [os.path.join(self.images_path, f) for f in os.listdir(self.images_path)]

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            executor.map(self.process_image, img_paths)

        self.df.to_csv(f'{self.to_path}/data.csv')

    
    #def create_threads(self):





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--images_path', default='data/unsplash')
    parser.add_argument('--to_path', default='data/generated')
    parser.add_argument('--threads', default=10)
    parser.add_argument('--var_per_img', default=10)
    parser.add_argument('--max_samples', default=None)


    args = parser.parse_args()

    generator = DataGenerator(args.images_path, args.to_path, args.var_per_image, args.threads, args.max_samples)
    generator.run()