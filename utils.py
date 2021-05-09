import os
from torch.utils.tensorboard import SummaryWriter
import tensorly as tl
import tensorly.decomposition as dc
from tqdm import tqdm
import torch.nn.utils.prune as prune
tl.set_backend('pytorch')
import torch

def create_summary_writer(model, data_loader, save_folder, model_id, device='cpu', conv=False):
    """Create a logger.

    Parameters
    ----------
    model
        Pytorch model.
    data_loader
        Pytorch DataLoader.
    save_folder: str
        Base location to save models and metadata.
    model_id: str
        Model/hp ID.

    Returns
    -------
    writer
        Logger object.
    """
    model.eval()
    writer = SummaryWriter(os.path.join(save_folder, model_id))
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    x = x.to(device)
    if conv:
        x = x.unsqueeze(1)
    with writer:
        try:
            writer.add_graph(model, x)
        except Exception as e:
            print("Failed to save model graph: {}".format(e))
    return writer

def select_filters(model, valid_loader, valid_set, remove_amount, device):
    """
    worst : list of highest divergence filters (worst filters) across batches
            Can select top-k afterwards.
    imp   : list of divergences from tensor decomposition reconstruction.
            lower means filter is more important.
    """
    worst = []
    model.eval()
    for i, data in tqdm(enumerate(valid_loader),
                        total=len(valid_set) / valid_loader.batch_size):
        out, y = data
        out = out.to(device)
        y = y
        for j, (name, param) in enumerate(model.named_children()):
            out = param(out)
            if j == 0:
                break
        nout = out.detach()

        cp = dc.tucker(nout, 15)
        pred = tl.tucker_tensor.tucker_to_tensor(cp)
        dist = torch.cdist(pred, nout)
        importance = torch.mean(dist, dim=[0, 2, 3])
        _, w = torch.topk(importance, remove_amount)
        worst.append(w)
        
        if i == (len(valid_set) // valid_loader.batch_size)//4:
            break
    return worst

class TuckerPruningMethod(prune.BasePruningMethod):
    def __init__(self, amount, dim=0, filt=0):
        self.amount = amount
        self.dim = dim
        self.filt = filt
    PRUNING_TYPE = 'structured'
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask[self.filt] = 0
        return mask

def TuckerStructured(module, name, amount=1, dim=8, filt=0):
    TuckerPruningMethod.apply(module, name, amount, dim, filt)
    return module