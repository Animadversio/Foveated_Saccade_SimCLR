import torch
from typing import Tuple, List, Optional
from packaging import version

torchver = version.parse(torch.__version__)
OLD_DIV = torchver < version.parse("1.8.0")


def unravel_indices(
  indices: torch.LongTensor,
  shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.
    Args:
    indices: A tensor of (flat) indices, (*, N).
    shape: The targeted shape, (D,).
    Returns:
    The unraveled coordinates, (*, N, D).
    """
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        if OLD_DIV:
            indices = indices // dim
        else:
            # use this version to suppress the torch warning about changed behavior in newer torch.
            indices = torch.div(indices, dim, rounding_mode="floor")
    coord = torch.stack(coord[::-1], dim=-1)
    return coord


def send_to_clipboard(image):
    """https://stackoverflow.com/questions/34322132/copy-image-to-clipboard"""
    from io import BytesIO
    import win32clipboard # this is not in minimal env.
    output = BytesIO()
    image.convert('RGB').save(output, 'BMP')
    data = output.getvalue()[14:]
    output.close()

    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
    win32clipboard.CloseClipboard()

