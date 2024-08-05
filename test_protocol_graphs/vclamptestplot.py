#test plot of voltage clamp data

import numpy as np
import matplotlib.pyplot as plt
import random

# open a file with voltage clamp data
with open('testvclampdata.csv', 'r') as f:
    data = f.readlines()
    real_numbers = [float(line.strip()) for line in data]

def plot_numbers(numbers):
    plt.figure(figsize=(10, 5))
    plt.plot(numbers, marker='o')  # 'o' creates a circle marker at each data point
    plt.title('Plot of Random Numbers')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

 # Using the function to plot the random numbers
plot_numbers(real_numbers)