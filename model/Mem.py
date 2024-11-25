import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryAugmented(nn.Module):
    def __init__(self, T, mem_num, mem_dim):
        super(MemoryAugmented, self).__init__()
        self.M = nn.Parameter(torch.FloatTensor(T, mem_num, mem_dim))
        nn.init.xavier_normal_(self.M)

    def forward(self, x):
        score = torch.softmax(torch.einsum("blnd,tmd->btnm", x, self.M), dim=-1)
        value = torch.einsum("blnm,tmd->btnd", score, self.M)
        return value


class STDMN(nn.Module):
    def __init__(self, T=12, num_nodes=170, model_dim=64, mem_num=10, mem_dim=128):
        super(STDMN, self).__init__()
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.T = T
        self.ma = MemoryAugmented(T, mem_num, mem_dim)
        self.l1 = nn.Linear(model_dim, mem_dim)
        self.l2 = nn.Linear(mem_dim, model_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.l1(x)
        x = self.ma(x)
        x = self.dropout(self.l2(x))
        return x


class MemoryAugmentedConv(nn.Module):
    def __init__(self, mem_num=40, mem_dim=64, dropout=0.2):
        super(MemoryAugmentedConv, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.M = nn.Parameter(torch.FloatTensor(mem_num, mem_dim))
        nn.init.xavier_normal_(self.M)

    def forward(self, x):
        score = torch.softmax(torch.einsum("btnd,md->btnm", x, self.M), dim=-1)
        value = torch.einsum("btnm,md->btnd", score, self.M)
        return self.dropout(value)


class MSAM(nn.Module):
    def __init__(self, input_dim, output_dim, mem_num=10, conv_stride=2, conv_k=3, dim_k=24):
        super(MSAM, self).__init__()
        self.dim_k = dim_k
        self.k = conv_k
        self.linear1 = nn.Linear(input_dim, self.dim_k)  # D -> D_k

        # MemoryAugmented
        self.mem_augs = nn.ModuleList([MemoryAugmentedConv(mem_num=mem_num, mem_dim=self.dim_k) for _ in range(self.k)])
        self.convs = nn.ModuleList([
            nn.Conv2d(self.dim_k, self.dim_k, kernel_size=((i+1), 1), stride=(conv_stride + i, 1)) for i in range(self.k)
        ])
        self.linear2 = nn.Linear(self.dim_k, input_dim)  # D_k -> D

    def forward(self, x):
        batch_size, T, N, D = x.size()  # B x T x N x D

        # D -> D_k
        x = self.linear1(x)  # B x T x N x D_k
        x = x.permute(0, 3, 1, 2)  # B x D_k x T x N
        conv_outputs = []

        for i, conv in enumerate(self.convs):
            x_conv = conv(x)  #
            x_conv = self.mem_augs[i](x_conv.permute(0, 2, 3, 1))
            conv_outputs.append(x_conv)

            # along T
        out = torch.cat(conv_outputs, dim=1)  # B x T x N x D_k

        #  D_k -> D
        out = self.linear2(out)

        return out


if __name__ == '__main__':
    conv_stride = 1
    model = MSAM(64, 64, conv_stride=conv_stride)
    x = torch.randn(64, 12, 170, 64)  # 输入维度 B x T x N x D
    xm = model(x)

    # 输出形状
    print(xm.shape)  # 拼接后的 memory 结果
    print("参数总数:", sum(p.numel() for p in model.parameters()))

    # mem = MemoryAugmented(mem_num=64, mem_dim=64)
    # x = torch.randn(32, 19, 170, 64)
    # y = mem(x)
    # print(y.shape)
