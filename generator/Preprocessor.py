# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# 2025/11/14
# Preprocessor.py

import os
import sys
import shutil
import cv2

from PIL import Image
import glob
import numpy as np

import traceback

class Preprocessor:

  def __init__(self):
    pass
    
  def run(self, root_dir, 
                        output_images_dir, output_masks_dir):
    tiles_dir = os.listdir(root_dir)
    index = 0
    for tile_dir in tiles_dir:
       fullpath = os.path.join(root_dir, tile_dir)
       if not  os.path.isdir(fullpath):
          continue
       index += 1
       
       print("fullpath {}".format(fullpath))
       image_files = sorted(glob.glob(fullpath + "/images/*.jpg"))
       self.combine(image_files, output_images_dir, index)   

       mask_files = sorted(glob.glob(fullpath  + "/masks/*.png"))
       self.combine(mask_files, output_masks_dir, index)   
       
  def combine(self, image_files, output_dir, index):
    image = cv2.imread(image_files[0])
    fh, fw, c = image.shape
    background = Image.new("RGB", (fw*3, fh*3), (0,0,0))
    i = 0
    h = 0
    w = 0
    x = 0
    y = 0
    for m in range(3):
      for n in range(3):
        print(image_files[i])
        image = Image.open(image_files[i])
        w, h = image.size
        background.paste(image, (x, y))
        x += w
        i+=1
      x = 0
      y += h

    output_filepath = os.path.join(output_dir, str(index) + ".png")
    background.save(output_filepath)


if __name__ == "__main__":
  try:
    root_dir = "./Semantic segmentation dataset/"
   
    output_dir        = "./Non-Tiled-Aerial-Imagery/"
    output_images_dir = "./Non-Tiled-Aerial-Imagery/images/"
    output_masks_dir  = "./Non-Tiled-Aerial-Imagery/masks/"

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    os.makedirs(output_images_dir)
    os.makedirs(output_masks_dir)


    processor = Preprocessor()
    processor.run(root_dir,
                  output_images_dir, 
                  output_masks_dir)
  except:
    traceback.print_exc()


