import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import kurtosis

import pandas as pd
import numpy as np
from fastprogress import progress_bar
from collections import defaultdict

from pdb import set_trace

from .utils import snr

@torch.no_grad()    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    act, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    corrects = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        corrects.append(correct[:k].any(dim=0).float())
    return pred, act, corrects, res
    
@torch.no_grad()
def validation(model, dataset, layer_names=None, topk=(1,5), map_output=None, batch_size=250, num_workers=10, 
               shuffle=False, pin_memory=True, mb=None):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(reduction='none')
    filepaths = [(os.path.sep).join(f.split(os.path.sep)[-2:]) for f,_ in dataset.imgs]
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                            shuffle=shuffle, pin_memory=pin_memory)
    
    model.eval()
    results = defaultdict(lambda: defaultdict(list))
    count = 0
    for i, batch in enumerate(progress_bar(dataloader, parent=mb)):
        batch_size = batch[0].shape[0]
        images = batch[0].to(device, non_blocking=True)
        target = batch[1].to(device, non_blocking=True)
        index = batch[2].tolist()
        filenames = [filepaths[idx] for idx in index]
        
        if layer_names is None:
            output = model(images)
            if map_output is not None:
                output = map_output(output)
            features = dict(output=output)
        else:
            with FeatureExtractor(model, layer_names) as extractor:
                features = extractor(images)
                
        for layer_name, output in features.items():
            loss = criterion(output, target)

            preds, act, accuracies, _ = accuracy(output, target, topk=topk)     

            # maximum target and non-target activation
            target_activations = torch.gather(output, 1, target.view(-1,1)).squeeze()
            mask = torch.ones_like(output, dtype=bool).scatter_(1, target.view(-1, 1), False)
            non_target_output = output[mask].view(output.shape[0], -1)
            max_nontarget_activation = non_target_output.max(dim=1).values
            snr = abs((target_activations-max_nontarget_activation) / math.sqrt(2))
            
            results[layer_name]['layer_name'] += [layer_name] * len(index)
            results[layer_name]['index'] += index            
            results[layer_name]['filenames'] += filenames
            results[layer_name]['target_label'] += target.tolist()        
            results[layer_name]['predicted_label'] += preds[0].tolist()
            results[layer_name]['top5_labels'] += preds.t().tolist()
            results[layer_name]['top5_outputs'] += act.tolist()
            results[layer_name]['target_act'] += target_activations.tolist()
            results[layer_name]['max_nontarget_act'] += max_nontarget_activation.tolist()
            results[layer_name]['snr'] += snr.tolist()
            results[layer_name]['loss'] += loss.tolist()

            for idx,k in enumerate(topk):
                results[layer_name][f'correct{k}'] += accuracies[idx].tolist()

    # full image-by-image results 
    dfs = dict()
    summary = dict()
    for layer_name,res in results.items():
        df = pd.DataFrame(res)
        dfs[layer_name] = df
    
        # leaderboard summary    
        summary[layer_name] = dict(
            top1=df.correct1.mean()*100,
            top5=df.correct5.mean()*100,
            snr_mean=df.snr.mean(),
            snr_sqrt_mean_squared=np.sqrt(df.snr).mean()**2,
            snr_min=df.snr.min(),
            snr_max=df.snr.max(),
            snr_kurtosis=kurtosis(df.snr),
            snr_sqrt_kurtosis=kurtosis(np.sqrt(df.snr))
        )
    
    return dfs, summary