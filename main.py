#!/usr/bin/env python3.6

import argparse
import sys
import warnings
from pathlib import Path
from functools import reduce
from operator import add, itemgetter
#from shutil import copytree, rmtree
from typing import Any, Callable, List, Tuple

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from networks import weights_init
from dataloaders import get_loaders
from utilities import map_
from utilities import dice_coef, dice_batch, tqdm_, haussdorf
from utilities import probs2one_hot, probs2class, crop
import os

warnings.filterwarnings("ignore")
#os.chdir('MorphNonlins')

def setup(args, n_class: int) -> Tuple[Any, Any, Any, List[Callable], List[float], Callable]:
    print(">>> Setting up")
    cpu: bool = args.cpu or not torch.cuda.is_available()
    device = torch.device("cpu") if cpu else torch.device("cuda")

    if args.weights:
        if cpu:
            net = torch.load(args.weights, map_location='cpu')
        else:
            net = torch.load(args.weights)
        print(f">>> Restored weights from {args.weights} successfully.")
    else:
        net_class = getattr(__import__('networks'), args.network)
        net = net_class(args.channels, n_class).to(device)
        net.apply(weights_init)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.l_rate, betas=(0.9, 0.99), amsgrad=False)

    print(args.losses)
    losses = eval(args.losses)
    loss_fns: List[Callable] = []
    for loss_name, loss_indices, _ in losses:  #('GeneralizedDice', {'idc': [0, 1]}, 1)
        loss_class = getattr(__import__('losses'), loss_name)
        loss_fns.append(loss_class(**loss_indices))

    loss_weights = map_(itemgetter(2), losses)

#    scheduler = getattr(__import__('scheduler'), args.scheduler)(**eval(args.scheduler_params))  #<- should we add scheduling?
    return net, optimizer, device, loss_fns, loss_weights #, scheduler


def do_epoch(mode: str, net: Any, device: Any, loader: DataLoader, epc: int,
             loss_fns: List[Callable], loss_weights: List[float], C: int, optimizer: Any = None, compute_haussdorf: bool = False) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert mode in ["train", "val"]
    metric_axis: List[int] = [i for i in range(1,C)]

    if mode == "train":
        net.train()
        desc = f">> Training   ({epc})"
    elif mode == "val":
        net.eval()
        desc = f">> Validation ({epc})"

    total_iteration, total_images = len(loader), len(loader.dataset)
    all_dices: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    batch_dices: Tensor = torch.zeros((total_iteration, C), dtype=torch.float32, device=device)
    loss_log: Tensor = torch.zeros((total_iteration), dtype=torch.float32, device=device)
    haussdorf_log: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)

    tq_iter = tqdm_(enumerate(loader), total=total_iteration, desc=desc)

    done = 0
    for j, data in tq_iter:
        #data = [e.to(device) for e in data]  # Move all tensors to device
        image, target = data[:2] #image could contain multiple imgs, like for deepmedic.
        image = [e.to(device) for e in image]
        target = target.to(device)
        usingBL = np.any([fn.name=="SurfaceLoss" for fn in loss_fns])
        assert (len(data)>2)==usingBL, (len(data), usingBL) #we also mustve got DT, to calc boundary loss


        B = len(image[0])

        # Reset gradients
        if optimizer:
            optimizer.zero_grad()

        # Forward
        pred_logits: Tensor = net(*image)
        pred_probs: Tensor = F.softmax(pred_logits, dim=1)
        #print(f"\npred_log=({torch.min(pred_logits).item():.3f}, {torch.max(pred_logits).item():.3f}), pred_prob=({torch.min(pred_probs).item():.3f}, {torch.max(pred_probs).item():.3f})")
        predicted_mask: Tensor = probs2one_hot(pred_probs.detach())  # Used only for dice computation

        #label = list of targets/dts to use when calc. loss. so should be DT for boundary loss, and GT for else.
        labels = [crop(data[-1].to(device), pred_logits.shape) if fn.name=="SurfaceLoss" else crop(target, pred_logits.shape) for fn in loss_fns]
        ziped = zip(loss_fns, labels, loss_weights)
        losses = [w * loss_fn(pred_probs, label) for loss_fn, label, w in ziped]
        loss = reduce(add, losses)
        assert loss.shape == (), loss.shape
        #print(f"Loss: {loss}")

        # Backward
        if optimizer:
            loss.backward()
            optimizer.step()

          #  z = [x.grad.data for x in net.parameters()]
          #  ma = np.max([torch.max(zz) for zz in z])
          #  mi = np.min([torch.min(zz) for zz in z])
          #  print(f"Grad range: {(ma, mi)}")

        # Compute and log metrics
        loss_log[j] = loss.detach()

        sm_slice = slice(done, done + B)  # Values only for current batch

        tgt = crop(target.detach(), predicted_mask.shape)
        dices: Tensor = dice_coef(predicted_mask, tgt)
        assert dices.shape == (B, C), (dices.shape, B, C)
        all_dices[sm_slice, ...] = dices

        if B > 1 and mode == "val":
            batch_dice: Tensor = dice_batch(predicted_mask, tgt)
            assert batch_dice.shape == (C,), (batch_dice.shape, B, C)
            batch_dices[j] = batch_dice

        if compute_haussdorf:
            haussdorf_res: Tensor = haussdorf(predicted_mask.detach(), tgt)
            assert haussdorf_res.shape == (B, C)
            haussdorf_log[sm_slice] = haussdorf_res

        # Logging
        big_slice = slice(0, done + B)  # Value for current and previous batches

        dsc_dict = {f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis}
        hauss_dict = {f"HD{n}": haussdorf_log[big_slice, n].mean() for n in metric_axis} if compute_haussdorf else {}
        batch_dict = {f"bDSC{n}": batch_dices[:j, n].mean() for n in metric_axis} if B > 1 and mode == "val" else {}

        mean_dict = {"DSC": all_dices[big_slice, metric_axis].mean(),
                     "HD": haussdorf_log[big_slice, metric_axis].mean()} if len(metric_axis) > 1 else {}

        stat_dict = {**dsc_dict, **hauss_dict, **mean_dict, **batch_dict,
                     "loss": loss_log[:j].mean()}
        nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items()}

        tq_iter.set_postfix(nice_dict)
        done += B

    print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))

    return loss_log, all_dices, haussdorf_log



def run(args: argparse.Namespace) -> None:
    n_class: int = args.n_class
    lr: float = args.l_rate
    metric_axis: List(int) = [i for i in range(1, n_class)] #we are interested in metrics for all classes except bckg.


    # Proper params
    savedir: str = args.workdir
    n_epoch: int = args.n_epoch

    net, optimizer, device, loss_fns, loss_weights = setup(args, n_class)

    usingBL = np.any([fn.name=="SurfaceLoss" for fn in loss_fns])
    train_loader, val_loader = get_loaders(args.network, args.datasets,
                                           args.batch_size, n_class,
                                           args.debug, usingBL)

    n_tra: int = len(train_loader.dataset)  # Number of images in dataset
    l_tra: int = len(train_loader)  # Number of iteration per epoch: different if batch_size > 1
    n_val: int = len(val_loader.dataset)
    l_val: int = len(val_loader)

    best_dice: Tensor = torch.zeros(1).to(device).type(torch.float32)
    best_epoch: int = 0
    metrics = {"val_dice": torch.zeros((n_epoch, n_val, n_class), device=device).type(torch.float32),
               "val_loss": torch.zeros((n_epoch, l_val), device=device).type(torch.float32),
               "tra_dice": torch.zeros((n_epoch, n_tra, n_class), device=device).type(torch.float32),
               "tra_loss": torch.zeros((n_epoch, l_tra), device=device).type(torch.float32)}
    if args.compute_haussdorf:
        metrics["val_haussdorf"] = torch.zeros((n_epoch, n_val, n_class), device=device).type(torch.float32)

    print(">>> Starting the training")
    for i in range(n_epoch):
        # Do training and validation loops
        tra_loss, tra_dice, _ = do_epoch("train", net, device, train_loader, i, loss_fns,
                                                         loss_weights, n_class,
                                                         optimizer=optimizer)

        with torch.no_grad():
            val_loss, val_dice, val_haussdorf = do_epoch("val", net, device, val_loader, i, loss_fns,
                                                                         loss_weights, n_class,
                                                                         compute_haussdorf=args.compute_haussdorf)

        # Sort and save the metrics
        for k in metrics:
            assert metrics[k][i].shape == eval(k).shape, (metrics[k][i].shape, eval(k).shape)
            metrics[k][i] = eval(k)

        for k, e in metrics.items():
            np.save(Path(savedir, f"{k}.npy"), e.cpu().numpy())

        df = pd.DataFrame({"tra_loss": metrics["tra_loss"].mean(dim=1).cpu().numpy(),  #this is not exact avg, if batches not all of the same sizes.
                           "val_loss": metrics["val_loss"].mean(dim=1).cpu().numpy(),
                           "tra_dice": metrics["tra_dice"][:, :, metric_axis].mean(dim=(1,2)).cpu().numpy(),
                           "val_dice": metrics["val_dice"][:, :, metric_axis].mean(dim=(1,2)).cpu().numpy()})
        df.to_csv(Path(savedir, args.csv), float_format="%.4f", index_label="epoch")

        # Save model if better
        current_dice: Tensor = val_dice[:, metric_axis].mean()
        if current_dice > best_dice:
            best_epoch = i
            best_dice = current_dice
            best_dice_per_class = val_dice.mean(axis=0)
            if args.compute_haussdorf:
                best_haussdorf = val_haussdorf[:, metric_axis].mean()
                best_haussdorf_per_class = val_haussdorf.mean(axis=0)

            torch.save(net, Path(savedir, "best.pkl"))

        #optimizer, loss_fns, loss_weights = scheduler(i, optimizer, loss_fns, loss_weights)

        if args.schedule and ((i+1) % (best_epoch + 20) == 0):  # Yeah, ugly but will clean that later
            for param_group in optimizer.param_groups:
                lr *= 0.5
                param_group['lr'] = lr
                print(f'> New learning Rate: {lr}')

    #Now we're done, let's print when the best epoch was, and what were the results there:
    maybehaus = f", HD={best_haussdorf:.3f}{best_haussdorf_per_class.data.cpu().numpy()}" if args.compute_haussdorf else ""
    print(f">> Training done. Best epoch: {best_epoch}\n Best val metrics: DSC={best_dice:.3f}{best_dice_per_class.data.cpu().numpy()}{maybehaus}")



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--losses", type=str, required=True, help="List of (loss_name, loss_params, weight)")
    parser.add_argument("--network", type=str, required=True, help="The network to use")
    parser.add_argument("--n_class", type=int, required=True)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--in_memory", action='store_true')
    parser.add_argument("--schedule", action='store_true') #whether to halve the lr every so often
    parser.add_argument("--compute_haussdorf", action='store_true')
    parser.add_argument("--group", action='store_true', help="Group the patient slices together for validation. \
        Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.")

    parser.add_argument('--n_epoch', nargs='?', type=int, default=200, help='# of the epochs')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4, help='Learning Rate')
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--weights", type=str, default='', help="Stored weights to restore")

    sys.argv = ['main.py', "--datasets=data/ISLES", '--batch_size=2', '--in_memory', '--l_rate=1e-3', '--schedule', '--compute_haussdorf',
            '--n_epoch=15', '--workdir=results', '--csv=metrics.csv', '--n_class=2', '--channels=5',
            '--network=UNet', "--losses=[('GeneralizedDice', {'idc': [0, 1]}, 1)]"] #, ('SurfaceLoss', {'idc': [1]}, 0.5)]"]

    args = parser.parse_args()

    print(args)

    return args


if __name__ == '__main__':
    run(get_args())
