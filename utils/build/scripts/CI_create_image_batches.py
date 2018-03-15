#!/usr/bin/env python2

# INTEL CONFIDENTIAL
# Copyright 2018 Intel Corporation
#
# The source code contained or described herein and all documents related to the source code ("Material") are owned by
# Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel
# or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted,
# transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual property right is granted to
# or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement,
# estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel
# in writing.

import os
import argparse
import struct


class image_batcher:
    def __init__(self, arr_images):
        self.images = arr_images.images
        self.images_count = len(self.images)
        self.batches = [1, 2, 4, 8, 10, 16, 32, 64, 128]
        self.x = arr_images.x
        self.y = arr_images.y

    def run(self):
        main_dir = ""
        if self.x != self.y:
            main_dir = str(self.x) + "x" + str(self.y)
        else:
            main_dir = str(self.x)
        self.__create_dir(main_dir)

        for batch in self.batches:
            dir = main_dir + os.path.sep + str(batch)
            self.__create_dir(dir)
            for i in range(batch):
                img_num = i % self.images_count
                with open(dir + os.path.sep + self.images[img_num].image_name[:-4] + "_" + str(i) + ".bmp", "wb") as f:
                    f.write(self.images[img_num].data)
        pass

    def __create_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        pass


class array_of_images:
    def __init__(self, size_x, size_y):
        self.x = size_x
        self.y = size_y
        self.images = []
        pass

    def add_image(self, image):
        if image.x == self.x and image.y == self.y:
            self.images.append(image)
        pass


class image:
    def __init__(self, image_name):
        self.image = open(image_name, "rb")
        self.data = None
        with open(image_name, 'rb') as f:
            self.data = bytearray(f.read())
        self.image_name = image_name
        self.x, self.y = self.__decode()
        pass

    def __decode(self):
        print('Name: ', self.image_name)
        print('Type:', self.image.read(2).decode())
        print('Size: %s' % struct.unpack('I', self.image.read(4)))
        print('Reserved 1: %s' % struct.unpack('H', self.image.read(2)))
        print('Reserved 2: %s' % struct.unpack('H', self.image.read(2)))
        print('Offset: %s' % struct.unpack('I', self.image.read(4)))
        print('DIB Header Size: %s' % struct.unpack('I', self.image.read(4)))
        x = struct.unpack('I', self.image.read(4))
        y = struct.unpack('I', self.image.read(4))
        print("Width: ", x)
        print("Height: ", y)
        return int(x[0]), int(y[0])


def append_array_wrapper(array, img):
    if len(array) != 0:
        for arr in array:
            if img.x == arr.x and img.y == arr.y:
                arr.add_image(img)
                return
    array.append(array_of_images(img.x, img.y))
    array[-1].add_image(img)


def main(): #'parse as many images as you want. example cmd line which will crate two folder (512 with 2 diffrent images and 228) with batches: python create_image_batches bear_512 plane_512 cat_227'
    parser = argparse.ArgumentParser(description='Create batch of images.')
    parser.add_argument('--images', metavar='N', nargs='+',
                        help='names of images')
    parser.add_argument("-i", "--info",
                        help="info about the script", action="store_true")

    args = parser.parse_args()

    if args.info:
        print("Createa batch of images.")
        print("Parse as many images as you want. example cmd line which will crate two folder (512 with 2 diffrent images and 228) with batches: python create_image_batches --images bear_512 plane_512 cat_227")
        exit(0)

    array_of_arrays_of_images = []

    for arg in args.images:
        if ".bmp" not in arg:
            arg += ".bmp"
        img = image(arg)
        append_array_wrapper(array_of_arrays_of_images, img)

    for arr in array_of_arrays_of_images:
        img_batcher = image_batcher(arr)
        img_batcher.run()


if __name__ == "__main__":
    main()
