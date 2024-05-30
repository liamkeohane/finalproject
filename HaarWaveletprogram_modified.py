# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:53:15 2024
@author: Markus Gerholm
"""
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
import time
import HaarWaveletsfunctions
from PIL import Image, UnidentifiedImageError
import os
os.chdir("/Users/figgeolsson/Downloads")

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
            plt.imshow(compressed_image, cmap='gray')
            plt.title("Compressed image")
            plt.show(block=True)
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
    # Matrix multiplication approach
    start_time_matrix = time.time()
    compressed_image_matrix = HaarWaveletsfunctions.HWT(A)
    end_time_matrix = time.time()

    # Direct transformation approach
    start_time_direct = time.time()
    A = A / 255.0  # Normalize 
    compressed_image_direct = HaarWaveletsfunctions.HWT_direct(A)
    end_time_direct = time.time()

    # Prepare to display four images: original, matrix approach, direct approach, and another placeholder if needed
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))  # Create 1 row, 4 columns of images

    # Display original image
    axs[0].imshow(A * 255, cmap='gray')  # Scale back for display
    axs[0].set_title('Original image')
    axs[0].axis('off')  # Hide axes for better visibility

    # Display matrix compressed image
    axs[1].imshow(compressed_image_matrix, cmap='gray')
    axs[1].set_title('Matrix approach')
    axs[1].axis('off')

    # Scaling and clipping for display for the direct method
    compressed_image_direct = np.clip(compressed_image_direct, 0, 1) * 255
    compressed_image_direct = compressed_image_direct.astype(np.uint8)

    # Display direct compressed image
    axs[2].imshow(compressed_image_direct, cmap='gray')
    axs[2].set_title('Direct approach')
    axs[2].axis('off')

    # Placeholder for a possible fourth image or reuse for additional information
    axs[3].text(0.5, 0.5, 'Additional Info', horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
    axs[3].axis('off')

    plt.tight_layout()
    plt.show(block=True)

    # Print time statistics
    print(f"Matrix multiplication time: {end_time_matrix - start_time_matrix:.6f} seconds")
    print(f"Direct transformation time: {end_time_direct - start_time_direct:.6f} seconds")
        elif choice == "5":
            break
program()
