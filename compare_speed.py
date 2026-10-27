import time 
import torch

a = torch.rand(10000, 10000)

b = torch.rand(10000, 10000)
# Primera multiplicación para preparar las librerías.
_ = a @ b

inicio = time.perf_counter()
result = a @ b
fin = time.perf_counter() -inicio

print(f"Tiempo de ejecución: {fin:.6f} segundos")
print(f"Resultado: {result}")


a_gpu = a.to("cuda")
b_gpu = b.to("cuda")


_ = a_gpu @ b_gpu
torch.cuda.synchronize()

inicio = time.perf_counter()

resultado_gpu = a_gpu @ b_gpu
torch.cuda.synchronize()

tiempo_gpu = time.perf_counter() - inicio

print("Dispositivo:", resultado_gpu.device)
print(f"Tiempo GPU: {tiempo_gpu * 1000:.2f} milisegundos")
print(f"CPU tardó {fin / tiempo_gpu:.2f} veces el tiempo de GPU")

