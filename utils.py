import os
from torch.utils.tensorboard import SummaryWriter
import tensorly as tl
import tensorly.decomposition as dc
from tqdm import tqdm
import torch.nn.utils.prune as prune
tl.set_backend('pytorch')
import torch
from torch.nn import Sequential, Conv2d
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

def select_filters(model, valid_loader, valid_set, remove_percent, device):
    """
    worst : list of highest divergence filters (worst filters) across batches
            Can select top-k afterwards.
    imp   : list of divergences from tensor decomposition reconstruction.
            lower means filter is more important.
    """
    worst = []
    bneck_layers = ['conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3', 'relu']
    model.eval()
    num_layers = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(valid_loader),
                            total=len(valid_set) / valid_loader.batch_size):
            out, y = data
            out = out.to(device)
            y = y
            sizes = []
            num_lay = 0
            for j, (name, param) in enumerate(model.named_children()):
                if name in ['avgpool', 'layer3']:
                    break
                if type(param) == Sequential:
                    for bottle in param:
                        for b in bneck_layers:
                            out = getattr(bottle, b)(out)
                            if b in ['conv1', 'conv2', 'conv3']:
                                nout = out.detach().clone()
                                num_rem = int(nout.shape[1] * remove_percent)
                                cp = dc.tucker(nout, 15)
                                pred = tl.tucker_tensor.tucker_to_tensor(cp)
                                dist = torch.cdist(pred, nout)
                                importance = torch.mean(dist, dim=[0, 2, 3])
                                _, w = torch.topk(importance, num_rem)
                                worst.append(w)
                                num_lay += 1

                else:
                    out = param(out)
                    if type(param) == Conv2d:
                        nout = out.detach().clone()
                        num_rem = int(nout.shape[1] * remove_percent)
                        cp = dc.tucker(nout, 15)
                        pred = tl.tucker_tensor.tucker_to_tensor(cp)
                        dist = torch.cdist(pred, nout)
                        importance = torch.mean(dist, dim=[0, 2, 3])
                        _, w = torch.topk(importance, num_rem)
                        worst.append(w)
                        num_lay += 1
            if i * valid_loader.batch_size >= 200:
                num_layers = num_lay
                break
    return worst, num_layers

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