class C2f_FADC(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_FADC(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

######################################## Frequency-Adaptive Dilated Convolution end ########################################

######################################## Parallelized Patch-Aware Attention Module start ########################################
