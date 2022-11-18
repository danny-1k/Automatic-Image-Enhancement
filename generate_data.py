import os
from models import ImageSelector
from data_utils import read_img, random_manipulation
import pandas as pd

import string
import random
from concurrent.futures import ThreadPoolExecutor


class DataGenerator:
    def __init__(self, images_path, to_path, var_per_image, threads , max_samples, efficient=True):
        self.efficient = efficient
        self.images_path = images_path
        self.to_path = to_path
        self.images_path = images_path
        self.max_samples = int(max_samples) if max_samples else max_samples
        self.threads = int(threads)
        self.num_sampled = 0


        self.var_per_image = var_per_image

        self.image_selector = ImageSelector(rho=0.2)


        self.df = pd.DataFrame({}, columns=['img_id', 'original_img_path', 'brightness', 'saturation', 'contrast', 'sharpness'], index=['img_id'])


    def process_image(self, f):
        if (self.max_samples and (self.num_sampled < self.max_samples)) or not self.max_samples:
            try:
                img = read_img(f)
                
                for _ in  range(self.var_per_image):

                    img_id = self.randid()

                    new_img, params = self.get_random_image(img)

                    if self.efficient:
                        new_img = new_img.resize((256, 256))

                    new_img.save(f'{self.to_path}/images/{img_id}.jpg')

                    self.df = self.df.append({
                        'img_id': img_id,
                        'original_img_path': f,
                        'brightness': params[0],
                        'saturation': params[1],
                        'contrast': params[2],
                        'sharpness': params[3]
                    }, ignore_index=True)

                    self.num_sampled += 1
                    
                    if self.num_sampled% 50 < 5:

                        self.save()
                        
                self.save()

                

            except:
                self.save()
                # self.save()
                # print('Stopped..')


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
            os.makedirs(f'{self.to_path}/images')

        else:
            assert os.listdir(f'{self.to_path}/images') == [], f'There are already Images in {self.to_path}/images'

        img_paths = [os.path.join(self.images_path, f) for f in os.listdir(self.images_path)]


        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            executor.map(self.process_image, img_paths)


        # for img in img_paths:
        #     self.process_image(img)
        #make dis shii fassstttttttttt
        # dont try multiprocessing, too many issues

        self.save()


    def randid(self):
        id = ''.join(random.choices(string.ascii_letters, k=10))
        return id

    def save(self):
        self.df.to_csv(f'{self.to_path}/data.csv')



    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--images_path', default='data/unsplash')
    parser.add_argument('--threads', default=1)
    parser.add_argument('--to_path', required=True)
    parser.add_argument('--var_per_img', default=10, type=int)
    parser.add_argument('--max_samples', default=None, type=int)


    args = parser.parse_args()

    generator = DataGenerator(args.images_path, args.to_path, args.var_per_img, args.threads, args.max_samples)
    generator.run()