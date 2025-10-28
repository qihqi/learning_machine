import torch
import jax

torch.Tensor.is_pinned = lambda *args, **kwargs: False

def main(get_torchax: bool = False):
    if get_torchax:
        import torchax
    x = torch.randn(1, 2, 3, 4, 5).to(torch.bfloat16)
    try:
        x_jax = jax.numpy.from_dlpack(x)
        print("Success")
    except RuntimeError as e:
        print("Failed")
        raise


if __name__ == "__main__":
    import fire
    fire.Fire(main)
