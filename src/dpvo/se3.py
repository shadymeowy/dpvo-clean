from .lietorch import SE3
import torch

# SE3
# __init__(data, device)
# .identity(shape, device)
# .inv(self)
# .__mul__(self, other)
# .exp(self, x)
# .log()
# .data

if True:
    class ProxySE3:
        def __init__(self, data):
            if isinstance(data, torch.Tensor):
                self.proxy = SE3(data)
            elif isinstance(data, SE3):
                self.proxy = data
            elif isinstance(data, ProxySE3):
                self.proxy = data.proxy
            else:
                raise ValueError("Unsupported data type")

        @staticmethod
        def Identity(shape, device=None):
            # print("ProxySE3.Identity", shape, device)
            return ProxySE3(SE3.Identity(shape, device=device))

        def inv(self):
            return ProxySE3(self.proxy.inv())

        def __mul__(self, other):
            if isinstance(other, ProxySE3):
                print("ProxySE3.__mul__", self.proxy.data.shape, other)
                return ProxySE3(self.proxy.__mul__(other.proxy))
            elif isinstance(other, torch.Tensor):
                print("ProxySE3.__mul__", self, self.proxy.data.shape, other.shape)
                assert other.shape[-1] == 4
                result = self.proxy.__mul__(other)
                if isinstance(result, SE3):
                    return ProxySE3(result)
                elif isinstance(result, torch.Tensor):
                    return result
                else:
                    raise ValueError
            else:
                raise ValueError

        def exp(self, x=None):
            if x is None:
                assert not isinstance(self, ProxySE3)
                return ProxySE3(SE3.exp(self))
            else:
                assert not isinstance(x, ProxySE3)
                return ProxySE3(self.proxy.exp(x))

        def log(self):
            return self.proxy.log()

        @property
        def data(self):
            return self.proxy.data

        @property
        def device(self):
            return self.proxy.device

        def __getitem__(self, index):
            # print("ProxySE3.__getitem__", index)
            return ProxySE3(self.proxy.__getitem__(index))

        def __setitem__(self, index, other):
            if isinstance(other, ProxySE3):
                self.proxy[index] = other.proxy
            elif isinstance(other, torch.Tensor):
                self.proxy[index] = other
            else:
                raise ValueError

        def __repr__(self):
            return f"ProxySE3({repr(self.proxy)})"
else:
    ProxySE3 = SE3
