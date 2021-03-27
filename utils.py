import os
from torch.utils.tensorboard import SummaryWriter


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
