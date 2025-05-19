import glob
import numpy as np
import os
from PIL import Image
from scipy import linalg
import torch
from sklearn.linear_model import LinearRegression
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale, Lambda
from tqdm import tqdm
from argparse import Namespace
import json
from evaluation.utils import download
import evaluation.inception as inception
import evaluation.image_metrics as image_metrics

ALLOWED_IMAGE_EXTENSIONS = ['jpg','JPG','jpeg','JPEG','png','PNG']
CKPT_URL = 'https://huggingface.co/matthias-wright/art_inception/resolve/main/art_inception.pth'

# Identity transform
Identity = Lambda(lambda x: x)

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transforms=None):
        self.paths = paths
        self.transforms = transforms
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        return self.transforms(img) if self.transforms else img


def get_image_paths(folder, sort=False):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"No such directory: {folder}")
    files = [os.path.join(folder,f) for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder,f)) and f.split('.')[-1] in ALLOWED_IMAGE_EXTENSIONS]
    return sorted(files) if sort else files


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    B,C,H,W = feat.size()
    x = feat.view(B,C,H*W)
    G = torch.bmm(x, x.transpose(1,2))
    return G/(C*H*W)


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1+np.eye(sigma1.shape[0])*eps).dot(sigma2+np.eye(sigma2.shape[0])*eps))
    if np.iscomplexobj(covmean): covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2*np.trace(covmean)


def load_inception(device):
    ckpt = download(CKPT_URL)
    state = torch.load(ckpt, map_location=device)
    model = inception.Inception3().to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def get_activations(paths, model, batch_size=50, num_workers=0):
    device = next(model.parameters()).device
    ds = ImagePathDataset(paths, transforms=Compose([Resize((512,512)), ToTensor()]))
    loader = torch.utils.data.DataLoader(ds, batch_size=min(batch_size,len(paths)),
                                         shuffle=False, drop_last=False,
                                         num_workers=num_workers)
    feats = np.zeros((len(paths),2048), dtype=np.float32)
    idx = 0
    for batch in tqdm(loader, total=len(paths), desc="Activations"):
        batch = batch.to(device)
        with torch.no_grad(): out = model(batch, return_features=True).cpu().numpy()
        feats[idx:idx+out.shape[0]] = out; idx+=out.shape[0]
    return feats


def compute_activation_statistics(paths, model, batch_size=50, num_workers=0):
    act = get_activations(paths, model, batch_size, num_workers)
    return act.mean(axis=0), np.cov(act, rowvar=False)


def compute_fid_grouped(path_stylized, path_style, batch_size=50, device='cpu', num_workers=0):
  
    device = torch.device('cuda') if device=='cuda' and torch.cuda.is_available() else torch.device('cpu')
    model = load_inception(device)

    style_files    = get_image_paths(path_style, sort=True)
    stylized_files = get_image_paths(path_stylized, sort=True)

    mu_style, sigma_style = compute_activation_statistics(style_files, model, batch_size, num_workers)
    from collections import defaultdict
    groups = defaultdict(list)
    for p in stylized_files:
        name = os.path.splitext(os.path.basename(p))[0]
        prefix = name.rsplit('_', 1)[0]
        groups[prefix].append(p)

    fids=[]=[]
    for variants in groups.values():
        mu_v, sigma_v = compute_activation_statistics(variants, model, batch_size, num_workers)
        fids.append(compute_frechet_distance(mu_style, sigma_style, mu_v, sigma_v))
    if not fids: raise ValueError("No stylized variants found.")
    return float(np.mean(fids))


def compute_content_distance_grouped(path_stylized, path_content,
                                     batch_size=50,
                                     content_metric='lpips',
                                     device='cpu',
                                     num_workers=0,
                                     gray = False):
    
    dev = torch.device('cuda') if device=='cuda' and torch.cuda.is_available() else torch.device('cpu')
    resize_and_gray = Grayscale() if gray else Identity
    transform = Compose([Resize((512,512)), resize_and_gray, ToTensor()])
    stylized = get_image_paths(path_stylized, sort=True)
    contents = get_image_paths(path_content,  sort=True)
    if content_metric == 'lpips':
        metric = image_metrics.LPIPS().to(dev)
    elif content_metric == 'vgg':
        metric = image_metrics.LPIPS_vgg().to(dev)
    elif content_metric == "patch_simi":
        metric = image_metrics.PatchSimi(device=dev).to(dev)
    from collections import defaultdict
    by_style = defaultdict(list)
    for p in stylized:
        name = os.path.splitext(os.path.basename(p))[0]
        if '_style_' not in name:
            continue
        style_key = name.split('_style_', 1)[1]   
        by_style[style_key].append(p)

    style_scores = []
    for style_key, variants in by_style.items():
        distances = []
        for sp in variants:
            base = os.path.splitext(os.path.basename(sp))[0]
            content_key = base.split('_style_',1)[0]  
            cf = next((c for c in contents
                       if os.path.splitext(os.path.basename(c))[0] == content_key),
                      None)
            if cf is None:
                raise ValueError(f"No content file for '{content_key}'")
            img_s = transform(Image.open(sp).convert('RGB')).unsqueeze(0).to(dev)
            img_c = transform(Image.open(cf).convert('RGB')).unsqueeze(0).to(dev)
            with torch.no_grad():
                d = metric(img_s, img_c)
            distances.append(d.item())

        if not distances:
            continue
        style_scores.append(np.mean(distances))

    if not style_scores:
        raise ValueError("No style groups found for content-distance computation.")

    return float(np.mean(style_scores))


def compute_art_fid_grouped(path_stylized, path_style, path_content,
                             batch_size=50, content_metric='lpips',
                             device='cpu', num_workers=0):
    device = torch.device('cuda') if device=='cuda' and torch.cuda.is_available() else torch.device('cpu')
    fid = compute_fid_grouped(path_stylized, path_style, batch_size, device, num_workers)
    cnt = compute_content_distance_grouped(path_stylized, path_content,
                                          batch_size, content_metric, device, num_workers, gray=False)
    gray_cnt = compute_content_distance_grouped(path_stylized, path_content,
                                                batch_size, content_metric, device, num_workers, gray=True)
    return (cnt+1)*(fid+1), fid, cnt, gray_cnt


def compute_cfsd_grouped(path_stylized, path_content, batch_size=50,
                         device='cpu', num_workers=0):

    return compute_content_distance_grouped(path_stylized, path_content,
                                            batch_size, 'patch_simi', device, num_workers)


def evaluate(content_path, style_path, styled_path,
             batch_size=50, content_metric='lpips',
             device='cpu', num_workers=0):
    device = torch.device('cuda') if device=='cuda' and torch.cuda.is_available() else torch.device('cpu')
    artfid, fid, lp, lp_g = compute_art_fid_grouped(
        styled_path, style_path, content_path,
        batch_size, content_metric, device, num_workers
    )
    print(f"ArtFID: {artfid:.4f}   FID: {fid:.4f}   {content_metric}: {lp:.4f}   Gray: {lp_g:.4f}")
    cfsd = compute_cfsd_grouped(styled_path, content_path,
                                 batch_size, device, num_workers)
    print(f"CFSD: {cfsd:.4f}")
    return artfid, fid, lp, lp_g, cfsd