import math
import os
import random
import numpy as np
import sys
import pickle
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

WIDTH = 1024
HEIGHT = 1024

def processImg(imagefile):

    img = Image.open(imagefile)

    img_array = np.array(img.convert('RGB'))

    #print(img_array.shape)

    return img_array


def resize_and_center_crop(image_path, target_size=1024):
    """
    Resize image so smaller dimension = target_size, then center crop.
    Preserves aspect ratio and doesn't distort.
    """
    img = Image.open(image_path)

    # Calculate resize dimensions (make smaller dimension = target_size)
    width, height = img.size
    if width < height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))

    # Resize
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Center crop to target_size x target_size
    left = (new_width - target_size) // 2
    top = (new_height - target_size) // 2
    right = left + target_size
    bottom = top + target_size

    img_cropped = img_resized.crop((left, top, right, bottom))

    return np.array(img_cropped)

def processFolder(inputfolder):
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    imagefolder = [f for f in os.listdir(inputfolder)
                   if os.path.splitext(f)[1].lower() in image_extensions]

    print(str(len(imagefolder)) + " images found")

    return imagefolder

def splitImage(N, image, minsize, maxlen):
    s = image.shape
    pieces = [[0, 0, s[0], s[1]]]

    while len(pieces) < N:
        p = max(pieces, key=lambda r: r[2] * r[3])
        split_pieces = split(p, minsize)
        pieces.remove(p)
        pieces.extend(split_pieces)

    return pieces

def split(piece, minsize):
    if piece[2] > piece[3]:
        #split by height
        splitpoint = random.randint(int(minsize*piece[2]), int((1-minsize)*piece[2]))
        part1 = [piece[0], piece[1], splitpoint, piece[3]]
        part2 = [piece[0]+splitpoint, piece[1], piece[2]-splitpoint, piece[3]]
        return [part1, part2]
    elif piece[2] < piece[3]:
        #split by width
        splitpoint = random.randint(int(minsize * piece[3]), int((1 - minsize) * piece[3]))
        part1 = [piece[0], piece[1], piece[2], splitpoint]
        part2 = [piece[0] , piece[1]+ splitpoint, piece[2] , piece[3]- splitpoint]
        return [part1, part2]
    else:
        f = random.randint(1, 10)
        if f <= 5:
            splitpoint = random.randint(int(minsize * piece[2]), int((1 - minsize) * piece[2]))
            part1 = [piece[0], piece[1], splitpoint, piece[3]]
            part2 = [piece[0] + splitpoint, piece[1], piece[2] - splitpoint, piece[3]]
            return [part1, part2]
        else:
            splitpoint = random.randint(int(minsize * piece[3]), int((1 - minsize) * piece[3]))
            part1 = [piece[0], piece[1], piece[2], splitpoint]
            part2 = [piece[0], piece[1] + splitpoint, piece[2], piece[3] - splitpoint]
            return [part1, part2]

def mapToImage(pieces, image):
    imgp = []
    for p in pieces:
        x, y, w, h = p
        imgp.append(image[x:x + w, y:y + h, :])
    return imgp

def splitEvenly(N, image):
    w = int(image.shape[0]/int(math.sqrt(N)))
    h = int(image.shape[1]/int(math.sqrt(N)))
    pieces = []
    for i in range(int(math.sqrt(N))):
        for j in range(int(math.sqrt(N))):
            pieces.append([i*w, j*h, w, h])
    return pieces

#I USED AI ON THIS FUNCTION AND ONLY THIS ONE BECAUSE I DIDN'T KNOW HOW TO VISUALIZE A NUMPY ARRAY SO BE SURE TO FIX THIS LATER
def reconstruct_with_borders(pieces, rectangles, border_width=3, border_color=(255, 0, 0)):
    """
    Reconstruct the full image with borders around each piece.

    Args:
        pieces: List of NumPy arrays (image pieces)
        rectangles: List of [x, y, width, height] for each piece
        border_width: Width of borders in pixels
        border_color: RGB tuple for border color
    """
    # Find full image dimensions
    max_x = max(r[0] + r[2] for r in rectangles)
    max_y = max(r[1] + r[3] for r in rectangles)

    # Create blank canvas
    canvas = np.zeros((max_y, max_x, 3), dtype=np.uint8)

    # Place each piece
    for piece, rect in zip(pieces, rectangles):
        x, y, width, height = rect
        canvas[x:x + width, y:y + height] = piece

    # Convert to PIL and draw borders
    img_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_pil)

    for rect in rectangles:
        x, y, width, height = rect
        draw.rectangle([y, x, y + height - 1, x + width - 1],
                       outline=border_color,
                       width=border_width)

    return img_pil

class puzzle():

    pieces = []

    def __init__(self, p):
        pieces = p


if __name__ == "__main__":
    img = None

    if sys.argv[1] == None:
        print("invalid argument")
        sys.exit(1)

    if sys.argv[1] == "image":
        if len(sys.argv) != 4:
            print("invalid number of arguments")
            sys.exit(1)
        img = resize_and_center_crop(sys.argv[2])
        splittedImage = splitImage(int(sys.argv[3]), img, .3, .5)

        mapped = mapToImage(splittedImage, img)

        result = reconstruct_with_borders(mapped, splittedImage)
        plt.figure(figsize=(10, 10))
        plt.imshow(result)
        plt.title(f'Reconstructed Image with {len(mapped)} Pieces')
        plt.axis('off')
        plt.show()
    else:
        img = processFolder(sys.argv[2])
        splits = []
        for s in img:
            i = resize_and_center_crop(sys.argv[2]+"/"+s)
            for n in range(16, 36):
                splittedImage = splitImage(n, i, .3, .5)
                mapped = mapToImage(splittedImage, i)
                splits.append([mapped, splittedImage])
                """
                result = reconstruct_with_borders(mapped, splittedImage)
                plt.figure(figsize=(10, 10))
                plt.imshow(result)
                plt.title(f'Reconstructed Image with {len(mapped)} Pieces')
                plt.axis('off')
                plt.show()"""
            sp = splitEvenly(4, i)
            m = mapToImage(sp, i)
            splits.append([sp, m])
            sp = splitEvenly(4, i)
            m = mapToImage(sp, i)
            splits.append([sp, m])

        print(str(len(splits)) + " images processed")

        with open('processed_images.pkl', 'wb') as f:
            pickle.dump(splits, f)
            f.close()
        print("data saved!")