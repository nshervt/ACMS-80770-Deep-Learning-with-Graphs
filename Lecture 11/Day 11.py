import torch
import torch.optim as optim

from torch import nn
"""x = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(x)
print(x.numpy())
print(x[1, 2].item())

x = torch.randn(4, 3)
print(x)"""



# -- Autograd
"""x = torch.tensor(5.0, requires_grad=True)
u = torch.tensor([1., 2.], requires_grad=True)


y = 3 * x**3 + torch.sin(x**2) + x * u[0] + u[1]

print(y)

y.backward()

print(x.grad)
print(u.grad)"""

# -- regression

"""# generate data
X = torch.randn(100, 10)
W0 = torch.randn(1, 10)
b0 = torch.randn(1, 1)

Y = W0@X.T+b0

# init
W = torch.randn(1, 10, requires_grad=True)
b = torch.randn(1, 1, requires_grad=True)

L, itr, eta = torch.tensor(1.), 0, 0.01

while L>0.001 and itr<1000:
    itr += 1

    Y_hat = W @ X.T + b

    L = torch.mean((Y_hat-Y)**2.0)

    L.backward()

    dLdW = W.grad
    dLdb = b.grad

    with torch.no_grad():
        W -= eta * dLdW
        b -= eta * dLdb

    W.grad.data.zero_()
    b.grad.data.zero_()

    print(L.item())"""

# -- Optimizer

"""# generate data
X = torch.randn(100, 10)
W0 = torch.randn(1, 10)
b0 = torch.randn(1, 1)

Y = W0@X.T+b0

# init
W = torch.randn(1, 10, requires_grad=True)
b = torch.randn(1, 1, requires_grad=True)

L, itr, eta = torch.tensor(1.), 0, 0.01

optmizer = optim.SGD([W, b], lr=0.01)

while L > 0.001 and itr < 1000:
    itr += 1

    Y_hat = W @ X.T + b

    L = torch.mean((Y_hat-Y)**2.0)

    optmizer.zero_grad()
    L.backward()
    # dLdW = W.grad
    # dLdb = b.grad
    optmizer.step()
    # with torch.no_grad():
    #     W -= eta * dLdW
    #     b -= eta * dLdb

    # W.grad.data.zero_()
    # b.grad.data.zero_()

    print(L.item())"""

# --

# generate data
X = torch.randn(100, 10)
W0 = torch.randn(1, 10)
b0 = torch.randn(1, 1)

Y = W0@X.T+b0

# init
model = nn.Linear(10, 1)

for param in model.parameters():
    print(param.shape)

optimizer = optim.SGD(model.parameters(), lr=0.01)

my_loss = nn.MSELoss()

L, itr = torch.tensor(10.), 0
while L > 0.001 and itr < 1000:
    itr += 1

    Y_hat = model(X)
    optimizer.zero_grad()
    L = my_loss(Y_hat, Y.T)
    L.backward()

    optimizer.step()

    print('loss: {}'.format(L.item()))
