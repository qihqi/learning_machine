from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch
import torchax
from torchax.interop import JittableModule
from torchvision import datasets, transforms

from torchax.interop import jax_view, call_torch

torchax.enable_globally()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, drop_last=True
)


class TorchPart(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class HybridModel(eqx.Module):
    jax_linear1: eqx.nn.Linear
    # torch_part: JittableModule
    jax_linear2: eqx.nn.Linear
    torch_weights: tuple
    _func_call_torch: Any


    def __init__(self, key):
        self.jax_linear1 = eqx.nn.Linear(784, 128, key=key)
        torch_model = TorchPart(128, 128).to("jax")
        torch_part = JittableModule(torch_model)
        # NOTE: jax_view makes a jax-pytree from a torch-pytree
        self.torch_weights = jax_view(torch_part.params)
        self.jax_linear2 = eqx.nn.Linear(128, 10, key=key)

        def _call_torch_functional(weights, x):
          # NOTE: buffer part is usually the non-trainable part of the model, so
          # here we pass in as closure and it wont change by training as equinox doesnt know it
          # If you want to train it then you can assign it to a attr like torch_weights, and
          # take it as input in this function
          return torch_part.functional_call('forward', weights, torch_part.buffers, x)
        self._func_call_torch = _call_torch_functional

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape(-1, 784)
        x = eqx.filter_vmap(self.jax_linear1)(x)
        x = x.reshape(-1, 128)
        x = call_torch(self._func_call_torch, self.torch_weights, x)
        x = jax.nn.relu(x)
        x = eqx.filter_vmap(self.jax_linear2)(x)
        return x


def loss_fn(model: HybridModel, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=y
    ).mean()


def torchax_filter(element: Any):
    if eqx.is_array(element):
        return True
    if isinstance(element, torch.Tensor):
        return True
    return False


@eqx.filter_jit
def train_step(
    model: HybridModel,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tuple[HybridModel, Any, jnp.ndarray]:
    (loss_val, grads) = eqx.filter_value_and_grad(loss_fn)(model, x, y)

    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


main_key = jax.random.key(42)
model = HybridModel(main_key)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, torchax_filter))

linear_before_training = model.torch_weights["linear.weight"]
print("before training", linear_before_training)


for epoch in range(1):
    for batch_idx, (x_torch, y_torch) in enumerate(train_loader):
        x_jax = jnp.array(x_torch.numpy())
        y_jax = jnp.array(y_torch.numpy())

        model, opt_state, loss = train_step(model, opt_state, optimizer, x_jax, y_jax)

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item()}")

        if batch_idx >= 500:
            break


linear_after_training = model.torch_weights["linear.weight"]
print("after training", linear_after_training)
