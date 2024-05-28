# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:31:52 2024
@author: Markus Gerholm
"""
import numpy as np

def HWT(A): # Function for compressing image after converting image to array
    M, N = A.shape # Different dimensions of array
    
    if M % 2 != 0: # Ensures even dimensions for the array
        A = A[:-1, :]  # Remove the last row
        M -= 1  # Update M
    if N % 2 != 0:
        A = A[:, :-1]  # Remove the last column
        N -= 1  # Update N
        
    W_M = np.zeros((M, M)) # The first Haar Wavelet transformation matrix based on M
    W_N = np.zeros((N, N)) # The second Haar Wavelet transformation matrix based on N
    num_rows_to_set1 = M // 2 
    num_rows_to_set2 = N // 2
   
    for i in range(num_rows_to_set1):
        start_index = 2 * i  # Start index for the non-zero values
        W_M[i, start_index:start_index + 2] = np.sqrt(2) / 2

    for i in range(num_rows_to_set1, M):
        start_index = 2 * (i - num_rows_to_set1)  # Start index for the non-zero values
        W_M[i, start_index:start_index + 2] = np.array([-np.sqrt(2) / 2, np.sqrt(2) / 2])
        
    for i in range(num_rows_to_set2):
        start_index = 2 * i  # Start index for the non-zero values
        W_N[i, start_index:start_index + 2] = np.sqrt(2) / 2

    for i in range(num_rows_to_set2, N):
        start_index = 2 * (i - num_rows_to_set2)  # Start index for the non-zero values
        W_N[i, start_index:start_index + 2] = np.array([-np.sqrt(2) / 2, np.sqrt(2) / 2])
        
    W_N_T = W_N.T # Transposing second matrix so that dimensions remain intact after transformation
    A_1 = np.dot(W_M, A) # Multiplication of first matrix and image array
    A_right = np.dot(A_1, W_N_T) # Matrix multiplication of transformed matrix
    return A_right

def HWT_inverse(A):
    M, N = A.shape
    
    if M % 2 != 0:
        A = A[:-1, :]  
        M -= 1  
    if N % 2 != 0:
        A = A[:, :-1]  
        N -= 1  
        
    W_M = np.zeros((M, M))
    W_N = np.zeros((N, N))
    num_rows_to_set1 = M // 2
    num_rows_to_set2 = N // 2
   
    for i in range(num_rows_to_set1):
        start_index = 2 * i  
        W_M[i, start_index:start_index + 2] = np.sqrt(2) / 2

    for i in range(num_rows_to_set1, M):
        start_index = 2 * (i - num_rows_to_set1)  
        W_M[i, start_index:start_index + 2] = np.array([-np.sqrt(2) / 2, np.sqrt(2) / 2])
        
    for i in range(num_rows_to_set2):
        start_index = 2 * i  
        W_N[i, start_index:start_index + 2] = np.sqrt(2) / 2

    for i in range(num_rows_to_set2, N):
        start_index = 2 * (i - num_rows_to_set2)  
        W_N[i, start_index:start_index + 2] = np.array([-np.sqrt(2) / 2, np.sqrt(2) / 2])
        
    W_N_T = W_N.T 
    A_1 = np.dot(W_M, A) 
    A_right = np.dot(A_1, W_N_T) 
    W_M_inverse = (2 / np.sqrt(2)) * W_M.T # Multiplying so that normalizing factor = 1 and transposing W_M
    W_N_inverse = (2 / np.sqrt(2)) * W_N_T.T # Repeat step above
    A_inverse1 = np.dot(W_M_inverse, A_right) # Reversing compression of image with inverted matrices
    A_inverse2 = np.dot(A_inverse1, W_N_inverse)
    return A_inverse2 
    
def HWT_enhanced(A):
    M, N = A.shape
    
    if M % 2 != 0:
        A = A[:-1, :]  
        M -= 1  
    if N % 2 != 0:
        A = A[:, :-1]  
        N -= 1 
        
    W_M = np.zeros((M, M))
    W_N = np.zeros((N, N))
    num_rows_to_set1 = M // 2
    num_rows_to_set2 = N // 2
   
    for i in range(num_rows_to_set1):
        start_index = 2 * i  
        W_M[i, start_index:start_index + 2] = np.sqrt(2) / 2

    for i in range(num_rows_to_set1, M):
        start_index = 2 * (i - num_rows_to_set1)  
        W_M[i, start_index:start_index + 2] = np.array([-np.sqrt(2) / 2, np.sqrt(2) / 2])
        
    for i in range(num_rows_to_set2):
        start_index = 2 * i  
        W_N[i, start_index:start_index + 2] = np.sqrt(2) / 2

    for i in range(num_rows_to_set2, N):
        start_index = 2 * (i - num_rows_to_set2)  
        W_N[i, start_index:start_index + 2] = np.array([-np.sqrt(2) / 2, np.sqrt(2) / 2])
        
    W_N_T = W_N.T 
    A_1 = np.dot(W_M, A) 
    A_right = np.dot(A_1, W_N_T) 
    A_enhanced = A_right[0:int(M/2), 0:int(N/2)] # Extracting upper left corner of matrix to obtain the compressed image
    return A_enhanced

def HWT_modified(A):
    M, N = A.shape
    
    if M % 2 != 0:
        A = A[:-1, :]  
        M -= 1  
    if N % 2 != 0:
        A = A[:, :-1]  
        N -= 1  
# Function for compression of image such as HWT(A) but without matrix multiplication