import matplotlib.image as mpimg
from PIL import Image
import os
from pathlib import Path

"""
Brent Martin 22/4/2016
Routines for manipulating images.
Not called by the other code, but may be a useful starting point
if you wish to perform other image manipulations in python.
"""


path = 'C:/Brent/DATA/Teaching/COSC401/Cacophony videos/Possums/002_possum/'

# TEMP_FILE = 'c:/brent/temp/image.png'
TEMP_FILE = '../resources/image.png'


def shrink(in_filename, out_filename, factor):
    """ Creates a shrunk version of the images 1/10th of the original size """
    img = mpimg.imread(in_filename)
    # Underlying raw PIL can only deal with png files, so convert it to png first
    mpimg.imsave(TEMP_FILE, img)
    mpimg.thumbnail(TEMP_FILE, out_filename, factor)


def shrink_images(in_folder, out_folder, factor):
    """
    Loops through a directory of image folders
    shrinking the image files within and saving to the destination folder
    """
    for directory in os.listdir(in_folder):
        print(directory)
        in_dir = in_folder + "/" + directory
        out_dir = out_folder + "/" + directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for file in os.listdir(in_dir):
            in_file = in_dir + "/" + file
            out_file = out_dir + "/" + file.split('.')[0] + ".png"
            print("   converting " + in_file + " to " + out_file + "...")
            shrink(in_file, out_file, factor)


def format_names(images_path):
    for video_name in os.listdir(images_path):
        video_path = images_path + "/" + video_name
        video_number = video_name.split("_")[0]

        for image_name in os.listdir(video_path):
            image_path = video_path + "/" + image_name

            parts = image_name.split("_")
            new_image_name = parts[0] + "_" + video_number + "_" + parts[1]

            new_image_path = video_path + "/" + new_image_name

            os.rename(image_path, new_image_path)


def prefix_files(images_path, prefix):
    for image_name in os.listdir(images_path):
        image_path = images_path + "/" + image_name

        prefixed_image_path = images_path + "/" + prefix + "_" + image_name
        os.rename(image_path, prefixed_image_path)
        # print(prefixed_image_path)


if __name__ == "__main__":
    # shrink_images('C:/Brent/DATA/Teaching/COSC401/Images_large', 'C:/Brent/DATA/Teaching/COSC401/Images_small')
    shrink_images('../resources/fullsize images', '../resources/halfsize images', 0.5)
