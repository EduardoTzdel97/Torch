import torch

device = "cuda"

x = torch.tensor([1., 2., 3., 4.], device=device)


peso = torch.tensor(0, device=device, dtype=torch.float32, requires_grad=True)
sesgo = torch.tensor(0, device=device, dtype=torch.float32, requires_grad=True)
ruido = 0.5 * torch.randn_like(x)

y_real = 2 * x + 1 + ruido
y_pred = peso * x + sesgo 

error = ((y_pred - y_real) ** 2).mean()

# Calculamos las pendientes del error.
error.backward()

print("Error inicial:", error.item())
print("Gradiente del peso:", peso.grad.item())
print("Gradiente del sesgo:", sesgo.grad.item())

tasa = 0.1

with torch.no_grad():
    peso -= tasa * peso.grad
    sesgo -= tasa * sesgo.grad

nueva_prediccion = peso * x + sesgo
nuevo_error = ((nueva_prediccion - y_real) ** 2).mean()

print("Nuevo peso:", peso.item())
print("Nuevo sesgo:", sesgo.item())
print("Nueva predicción:", nueva_prediccion.detach().cpu().numpy())
print("Nuevo error:", nuevo_error.item())



tasa = 0.1
max_pasos = 1000
tolerancia = 1e-6

for paso in range(max_pasos):
    peso.grad = None
    sesgo.grad = None

    y_pred = peso * x + sesgo
    error = ((y_pred - y_real) ** 2).mean()

    error.backward()

    mayor_gradiente = max(
        abs(peso.grad.item()),
        abs(sesgo.grad.item()),
    )

    if paso % 100 == 0:
        print(
            f"Paso {paso}: error={error.item():.6f}, "
            f"gradiente máximo={mayor_gradiente:.6f}"
        )

    if mayor_gradiente < 1e-4:
        print(f"Convergencia alcanzada en el paso {paso}.")
        break

    with torch.no_grad():
        peso -= tasa * peso.grad
        sesgo -= tasa * sesgo.grad

else:
    print("Se alcanzó el máximo de pasos.")

print("Peso final:", peso.item())
print("Sesgo final:", sesgo.item())
print("Predicción final:", (peso * x + sesgo).detach().cpu().numpy())
print("Error final:", error.item())
print("Valores reales:", y_real.detach().cpu().numpy())


x_nuevos = torch.tensor(
    [0., 2.5, 5., 10.],
    device=device,
)

with torch.no_grad():
    predicciones = peso * x_nuevos + sesgo

print("Entradas nuevas:", x_nuevos.cpu().numpy())
print("Predicciones:", predicciones.cpu().numpy())