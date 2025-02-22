class DSConv(nn.Module):
    """Depthwise Separable Convolution"""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True) -> None:
        super().__init__()

        self.dwconv = DWConv(c1, c1, 3)
        self.pwconv = Conv(c1, c2, 1)

    def forward(self, x):
        return self.pwconv(self.dwconv(x))
