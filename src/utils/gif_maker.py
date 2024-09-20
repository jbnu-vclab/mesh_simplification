from PIL import Image
import wandb
from glob import glob

def images_to_gif(images_path, fname, duration, out_folder):
    image_fnames = glob(f'{images_path}/*.png')
    image_fnames.sort() #sort by step
    frames = [Image.open(image) for image in image_fnames]
    frame_one = frames[0]
    frame_one.save(f'{out_folder}/{fname}.gif', format="GIF", append_images=frames,
               save_all=True, duration=duration, loop=0)

if __name__=='__main__':
    images_to_gif('./gif_tmp', 'gif_test', 1, '.')