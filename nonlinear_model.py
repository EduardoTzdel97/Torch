import torch
import matplotlib.pyplot as plt

device = "cuda"

x = torch.tensor([-2., -1.,0, 1 ,2, ], device=device)

torch.manual_seed(0)  # Para reproducibilidad
y_real = x ** 2 + 3 * x + 5 + 0.5 * torch.randn_like(x)


a = torch.tensor(0, device=device, dtype=torch.float32, requires_grad=True)
b = torch.tensor(0, device=device, dtype=torch.float32, requires_grad=True)
c = torch.tensor(0, device=device, dtype=torch.float32, requires_grad=True)

y_pred = a * x ** 2 + b * x + c

error = ((y_pred - y_real) ** 2).mean()

error.backward()

print("Gradientes:", a.grad.item(), b.grad.item(), c.grad.item())
print("Predicción inicial:", y_pred.detach().cpu().numpy())
print("Error inicial:", error.item())


max_pasos = 100000
tolerancia = 1e-6
tasa = 0.00001

for pasos in range(max_pasos):
    a.grad = None
    b.grad = None
    c.grad = None

    y_pred = a * x ** 2 + b * x + c
    error = ((y_pred - y_real) ** 2).mean()

    error.backward()

    mayor_gradiente = max(
        abs(a.grad.item()),
        abs(b.grad.item()),
        abs(c.grad.item()),
    )

    if pasos % 100 == 0:
        print(
            f"Paso {pasos}: error={error.item():.6f}, "
            f"gradientes=({a.grad.item():.6f}, {b.grad.item():.6f}, {c.grad.item():.6f})"
        )

    if mayor_gradiente < tolerancia:
        print(f"Convergencia alcanzada en el paso {pasos}.")
        break

    with torch.no_grad():
        a -= tasa * a.grad
        b -= tasa * b.grad
        c -= tasa * c.grad


print("Predicción final:", (a * x ** 2 + b * x + c).detach().cpu().numpy())
print("Error final:", error.item())
print("Valores reales:", y_real.detach().cpu().numpy())
print("Coeficientes finales: a =", a.item(), ", b =", b.item(), ", c =", c.item())


x_curva=torch.linspace(x.min().item(), x.max().item(), 100, device=device)

with torch.no_grad():
    y_curva = a * x_curva ** 2 + b * x_curva + c
    y_real_curva = x_curva ** 2 + 3 * x_curva + 5

plt.scatter(
    x.detach().cpu().numpy(),
    y_real.detach().cpu().numpy(),
    label="Observaciones con ruido",
    alpha=0.6,
)

plt.plot(
    x_curva.cpu().numpy(),
    y_curva.cpu().numpy(),
    color="red",
    label="Curva aprendida",
)

plt.plot(
    x_curva.cpu().numpy(),
    y_real_curva.cpu().numpy(),
    color="black",
    linestyle="--",
    label="Curva real sin ruido",
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Modelo cuadrático: observaciones y predicción")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
