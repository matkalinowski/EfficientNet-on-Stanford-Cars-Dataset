from dataclasses import dataclass
from typing import List

from models.efficient_net.network_params import NetworkParams


@dataclass
class EfficientNetInfo:
    name: str
    network_params: NetworkParams
    pretrained_url: str
    advprop_pretrained_src: str
    block_args: List[str] = None

    def __post_init__(self):
        if not self.block_args:
            self.block_args = [
                'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
                'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
                'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
                'r1_k3_s11_e6_i192_o320_se0.25',
            ]

    def get_pretrained_url(self, advprop):
        if advprop:
            return self.advprop_pretrained_src
        else:
            return self.pretrained_url