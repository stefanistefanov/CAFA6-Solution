import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch

from cafa.core.cli import CLI


def main():
    torch.set_float32_matmul_precision('medium')
    CLI(
        parser_kwargs={'parser_mode': 'omegaconf'},
    )


if __name__ == '__main__':
    main()
