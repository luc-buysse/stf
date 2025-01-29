# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import random
import shutil
import sys
import os
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from compressai.models.stf import Adapter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP 

from torch.utils.tensorboard import SummaryWriter
import wandb

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target, epoch):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
  
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        out["z_loss"] = torch.log(output["likelihoods"]["z"]).sum()
        out["y_loss"] = torch.log(output["likelihoods"]["y"]).sum()

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(DDP):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = []
    aux_parameters = [p for n, p in net.named_parameters() if n.endswith(".quantiles")]

    for m in net.modules():
        if isinstance(m, Adapter):
            for p in m.parameters():
                parameters.append(p)
    
    if args.unfreeze_encoder:
        for layer in net.layers:
            for p in layer.parameters():
                parameters.append(p)

    optimizer = optim.Adam(
        parameters,
        lr=args.learning_rate,
        capturable=True,
    )
    aux_optimizer = optim.Adam(
        aux_parameters,
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm,
):
    model.train()
    device = next(model.parameters()).device
    n_items = len(train_dataloader)

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d, epoch)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0 and rank == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item() * 255 ** 2 / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )

            log({
                "Epoch": epoch,
                "Train / Loss": out_criterion['loss'].item(),
                "Train / MSE Loss": out_criterion['mse_loss'],
                "Train / BPP Loss": out_criterion['bpp_loss'],
                "Train / Aux loss": aux_loss.item(),
                "Train / z loss": out_criterion["z_loss"],
                "Train / y loss": out_criterion["y_loss"],
            })
            


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    z_loss = AverageMeter()
    y_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d, epoch)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

            z_loss.update(out_criterion["z_loss"])
            y_loss.update(out_criterion["y_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    log({
        "Epoch": epoch,
        "Test / Loss": loss.avg,
        "Test / MSE Loss": mse_loss.avg * 255 ** 2 / 3,
        "Test / Bpp Loss": bpp_loss.avg,
        "Test / Aux Loss": aux_loss.avg,
    })
    return loss.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-8]+"_best"+filename[-8:])


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Training configuration file path"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="stf",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=10,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="ckpt/model.pth.tar", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--eval",
        default=None,
        help="Additional evaluation set"
    )
    parser.add_argument(
        "--monitor",
        choices=['tensorboard', 'wandb', 'none'],
        default="tensorboard",
        help="Monitor: wandb / tensorboard / none"
    )
    parser.add_argument(
        "--name",
        help="Name of the run"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--limit", type=int, default=-1, help="A limit on the number of images to use for training (-1 for unlimited)")
    parser.add_argument("--unfreeze-encoder", action="store_true", help="Unfreeze all encoder parameters (no adapters)")
    args = parser.parse_args(argv)
    return args

def init_slurm():
    print('Trying to initialize SLURM with CUDA_VISIBLE_DEVICES set to : ', os.environ['CUDA_VISIBLE_DEVICES'])

    global rank, world_size
    if "WORLD_SIZE" in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['SLURM_PROCID'])
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(os.environ['SLURM_LOCALID'])

        dist.init_process_group(backend="nccl", init_method="env://",
            world_size=world_size, rank=rank)

def load_config(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    args.model = "stf"
    args.dataset = config['training']['dataset']
    args.save_path = config['training']['save']
    args.checkpoint = config['training']['original']
    args.eval = config['training']['eval']
    args.epochs = config['training']['epochs']
    args.learning_rate = config['training']['lr']
    args.lmbda = config['training']['lambda']
    args.cuda = True
    args.monitor = config['monitor']['type']
    args.name = config['monitor']['name']

    if 'limit' in config['training']:
        args.limit = config['training']['limit']
    else:
        args.limit = -1

    if "model" in config and 'encoder' in config['model'] and 'unfreeze' in config['model']['encoder']:
        args.unfreeze_encoder = config['model']['encoder']['unfreeze']

    return None if 'model' not in config else config['model']

def init_monitor(args):
    global monitor_type
    monitor_type = args.monitor

    if args.monitor == "tensorboard":
        global writer, writer_step
        writer = SummaryWriter(f'alice/{args.name}')
        writer_step = 0
    
    if args.monitor == "wandb":
        wandb.init(f'alice/{args.name}')

def log(content):
    global writer_step
    if monitor_type == "wandb":
        wandb.log(content)
    
    elif monitor_type == "tensorboard":
        for k, v in content.items():
            writer.add_scalar(k, v, writer_step)
            writer_step += 1


def main(argv):
    print(
        "Rank=", os.getenv("SLURM_PROCID", "?"), 
        "LocalRank=", os.getenv("SLURM_LOCALID", "?"), 
        "CUDA_VISIBLE_DEVICES=", os.getenv("CUDA_VISIBLE_DEVICES", "None"),
        flush=True
    )

    init_slurm()

    args = parse_args(argv)

    if args.config != None:
        model_config = load_config(args)
    
    if rank == 0:
        print(args)
    
    init_monitor(args)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)

    if args.limit != -1 and args.limit < len(train_dataset):
        subset_indices = range(0, args.limit)
        train_dataset = Subset(train_dataset, subset_indices)

    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=(device == "cuda"),
    )

    if rank == 0:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device == "cuda"),
        )

    if args.config != None:
        net = models[args.model](config=model_config)
    else:
        net = models[args.model]()

    net = net.to(device)

    has_data_parallel = False
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        has_data_parallel = True

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        if rank == 0:
            print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)

        if "epoch" in checkpoint:
            last_epoch = checkpoint["epoch"] + 1
        
        # If checkpoint has DataParallel and model doesn't
        if not has_data_parallel and next(iter(checkpoint["state_dict"].keys()))[:7] == "module.":
            checkpoint["state_dict"] = {k[7:]: v for k, v in checkpoint["state_dict"].items() if k[:7] == "module."}
        # If checkpoint doesn't have DataParallel and model does
        if has_data_parallel and next(iter(checkpoint["state_dict"].keys()))[:7] != "module.":
            checkpoint["state_dict"] = {f"module.{k}": v for k, v in checkpoint["state_dict"].items()}

        net.load_state_dict(checkpoint["state_dict"], strict=False)

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "aux_optimizer" in checkpoint:
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    loss = 0
    for epoch in range(last_epoch, args.epochs):
        if rank == 0:
            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )

        if rank == 0:
            loss = test_epoch(epoch, test_dataloader, net, criterion)
            lr_scheduler.step(loss)

        if rank == 0:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "config": model_config,
                    },
                    is_best,
                    args.save_path,
                )
    
    dist.destroy_process_group()

    if monitor_type == "tensorboard":
        writer.close()

if __name__ == "__main__":
    main(sys.argv[1:])
