import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class EncoderLayer(nn.Module):

    def __init__():
        super().__init__()

        self._size = 0

    def size() ->int:
        return self._size


    def predict_length_from_mem(self):
        return 0


class DecoderLayer(nn.Module):

    def __init__():
        super().__init__()
        self._size = 0

    def size() ->int:
        return self._size


class Encoder(nn.Module):
    """Encoder.
    abstract class for Encoder
    """


    def __init__(self):
        super().__init__()

    def forward(self,
               x: torch.Tensor,
               mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encode(x, mask)


    def encode(self,
               x: torch.Tensor,
               mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

    def reparameters(self,
                     mean: torch.Tensor,
                     logv: torch.Tensor,
                     scale: float = 1.0) -> torch.Tensor:
        pass

    def predict_property(self,
                         name: str,
                         mem: torch.Tensor) -> torch.Tensor:
        pass


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()


    def decode(self,
               x: torch.Tensor,
               mem: torch.Tensor,
               src_mask: torch.Tensor,
               tgt_mask: torch.Tensor) -> torch.Tensor:
        pass


    def forward(self,
               x: torch.Tensor,
               mem: torch.Tensor,
               src_mask: torch.Tensor,
               tgt_mask: torch.Tensor) -> torch.Tensor:
        return self.decode(x, mem, src_mask, tgt_mask)

