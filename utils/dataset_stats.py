from typing import Iterable, Sequence
import torch


def get_mean_std(data: Sequence[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    n_channels = data[0][0].shape[0]
    n_pixels = 0
    sum_channels = torch.zeros(n_channels)
    sum_sq_channels = torch.zeros(n_channels)

    for img, _ in data:
        n = img.shape[1] * img.shape[2]  # H*W
        n_pixels += n
        sum_channels += img.sum(dim=[1, 2])
        sum_sq_channels += (img**2).sum(dim=[1, 2])

    mean = sum_channels / n_pixels
    std = (sum_sq_channels / n_pixels - mean**2).sqrt()

    return mean, std


def get_min_max_sizes(
    data: Iterable[tuple[torch.Tensor, int]],
) -> tuple[int, int, int, int]:
    min_height: int = 10_0000
    min_width = 10_0000
    max_height = -1
    max_width = -1

    for t, _ in data:
        _, height, width = t.shape
        if height < min_height:
            min_height = height
        if width < min_width:
            min_width = width
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width

    return min_height, min_width, max_height, max_width


def summarize_stats(data: Iterable[tuple[torch.Tensor, int]]) -> str:
    min_height, min_width, max_height, max_width = get_min_max_sizes(data)
    mean, std = get_mean_std(data)

    s = f"Number of elements: {len(data)}\n"
    s += f"Minimum height and width: {min_height, min_width}, Maximum height and width: {max_height, max_width}\n"
    s += f"Mean: {mean}, standard deviation: {std}\n"

    return s


if __name__ == "__main__":
    import random

    torch.set_printoptions(precision=2)

    data = [(torch.rand(1, 224, 224), random.choice([0, 1])) for i in torch.arange(256)]

    print(summarize_stats(data))
