import torch
from torch.autograd import Function
import Swin3D.sparse_dl.pointops_cuda as pointops_cuda

class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset):
        """
        input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        """
        if new_xyz is None: new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        pointops_cuda.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)
        return idx, torch.sqrt(dist2)

knnquery = KNNQuery.apply
def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True, return_indx=False):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
    output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample)

    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) # (m, nsample, 3)
    #grouped_xyz = grouping(xyz, idx) # (m, nsample, 3)
    # 相对位置
    grouped_xyz -= new_xyz.unsqueeze(1) # (m, nsample, 3)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c) # (m, nsample, c)
    #grouped_feat = grouping(feat, idx) # (m, nsample, c)
    if use_xyz:
        if return_indx:
            return torch.cat((grouped_xyz, grouped_feat), -1), idx # (m, nsample, 3+c)
        else:
            return torch.cat((grouped_xyz, grouped_feat), -1)
    else:
        if return_indx:
            return grouped_feat, idx
        else:
            return grouped_feat

def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    """
    input: xyz: (m, 3), new_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset) # (n, 3), (n, 3)
    dist_recip = 1.0 / (dist + 1e-8) # (n, 3)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm # (n, 3)

    new_feat = torch.cuda.FloatTensor(new_xyz.shape[0], feat.shape[1]).zero_()
    for i in range(k):
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return new_feat
