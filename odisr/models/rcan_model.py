import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from torch.nn import functional as F

@MODEL_REGISTRY.register()
class RCANModel(SRModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(RCANModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'LAMB':
            import torch_optimizer
            optimizer = torch_optimizer.Lamb(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        return optimizer

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = test_TLC(self.lq, self.net_g, tile=64, tile_overlap=32, scale=self.opt['scale'])
        self.net_g.train()

def test_TLC(img_lq, model, tile=None, tile_overlap=32, scale=1):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
        if isinstance(output, list):
            output = output[-1]
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*scale, w*scale).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                if isinstance(out_patch, list):
                    out_patch = out_patch[-1]
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch)
                W[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch_mask)
        output = E.div_(W)

    return output