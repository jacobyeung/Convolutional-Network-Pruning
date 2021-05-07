import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tensorly as tl
import tensorly.decomposition as dc
from tqdm import tqdm
tl.set_backend('pytorch')

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

def select_filters(model, valid_loader, valid_set):
    """
    worst : list of highest divergence filters (worst filters) across batches
            Can select top-k afterwards.
    """
    worst = []
    for i, data in tqdm(enumerate(valid_loader),
                        total=len(valid_set) / valid_loader.batch_size):
        out, y = data
        out = out.to(device)
        y = y
        for i, param in enumerate(model.children()):
            out = param(out)
            if i == 0:
                break
#         out = model(out)
        nout = out.detach()

#         ny = y.detach().numpy()
#         for j in range(out.shape[-1]//10, out.shape[-1], out.shape[-1]//4):
#             print(j)
        cp = dc.tucker(nout, 25)
        pred = tl.tucker_tensor.tucker_to_tensor(cp)
        dist = torch.cdist(pred, nout)
        importance = torch.mean(dist, dim=[0, 2, 3])
        w = torch.argmax(importance)
        worst.append(w)
    return worst