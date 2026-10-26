""" import torch 
numbers = torch.tensor([1, 2, 3, 4, 5])

print(numbers)
print(numbers.shape)
print(numbers * 2) """

import torch
import matplotlib.pyplot as plt

""" image = torch.tensor([[0.0, 1.0, 0.0],
                      [1.0, 1.0, 1.0],
                      [0.0, 1.0, 0.0]])
darker = image * 0.5  # Darken the image by multiplying by 0.5

inverted = 1.0 - image  # Invert the image by subtracting from 1.0



print(image)
print("Shape of the image:", image.shape)

print(darker)
print("Shape of the darker image:", darker.shape)


plt.imshow(inverted.numpy(), cmap="gray", vmin=0, vmax=1)
plt.axis('off')  # Turn off axis labels
plt.show()

print("Centro" , image[1, 1])  # Access the center pixel value

image[1, 1] = 0.0  # Change the center pixel value to 0.0
print("Updated image after changing center pixel:")
plt.imshow(image.numpy(), cmap="gray", vmin=0, vmax=1)
plt.axis('off')  # Turn off axis labels
plt.show() """


a =torch.tensor([[1.0, 2.0],
                  [4.0, 5.0]])

print("Ubicacion inicial de a:", a.device)

a_gpu = a.to("cuda")  # Move the tensor to GPU
print("Ubicacion de a en GPU:", a_gpu.device)   

result = a_gpu @ a_gpu  # Perform operations on the GPU tenso

torch.cuda.synchronize()  # Wait for GPU operations to finish

print("Resultado de la multiplicacion en GPU:", result)


result_cpu = result.to("cpu")  # Move the result back to CPU
print("Resultado de la multiplicacion en CPU:", result_cpu)