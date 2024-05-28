# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:53:15 2024
@author: Markus Gerholm
"""
import numpy as np
import matplotlib.pyplot as plt
import HaarWaveletsfunctions
from PIL import Image, UnidentifiedImageError

def program():
    while True:
        start = input("Do you wish to compress an image? (yes/no) ")
        if start.startswith("y") !=True:
            break
        image_name = input("Name of file: ")
        try:
            image = Image.open(image_name).convert("L")
            A = np.asarray(image)
            compressed_image = HaarWaveletsfunctions.HWT(A)
            plt.imshow(compressed_image, cmap="gray")
            plt.title("Compressed image")
            plt.show()
        except FileNotFoundError:
            raise FileNotFoundError(f"'{image_name}' not found, ensure that the image is saved in the right folder and that the name of the file is correct")
        except UnidentifiedImageError:
            raise UnidentifiedImageError(f"'{image_name}' needs to be jpg or jpeg")
        print("1: Invert transformation process ")
        print("2: Enhance compressed image ")
        print("3: Save image to folder ")
        print("4: Compare computational time with modified code (work in progress) ")
        print("5: Quit program ")
        choice = input("Write your choice: ")
        if choice == "1":
            inverted_image = HaarWaveletsfunctions.HWT_inverse(A)
            plt.imshow(inverted_image, cmap="gray")
            plt.title('Inverted image')
            plt.show()
        elif choice == "2":
            enhanced_image = HaarWaveletsfunctions.HWT_enhanced(A)
            plt.imshow(enhanced_image, cmap='gray')
            plt.title('Compressed image, enhanced')
            plt.show()
            print("1: Compress image again ")
            print("2: Quit program")
            choice = input("Write your choice: ")
            if choice == "1":
                name = input("Name compressed image you wish to compress again: ")
                enhanced_image = np.clip(enhanced_image, 0, 255)
                enhanced_image = enhanced_image.astype(np.uint8)
                new_image = Image.fromarray(enhanced_image)
                new_image.save(name)
            elif choice == "2":
                break
        elif choice == "3":
            compressed_image = np.clip(compressed_image, 0, 255)
            compressed_image = compressed_image.astype(np.uint8)
            new_image = Image.fromarray(compressed_image)
            new_image.save("compressed_" + image_name)
        elif choice == "4":
            HaarWaveletsfunctions.HWT_modified(A) 
        elif choice == "5":
            break
program()