import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
from fastprogress import progress_bar
from collections import defaultdict

from pdb import set_trace

@torch.no_grad()    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    corrects = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        corrects.append(correct[:k].any(dim=0).float())
    return pred, corrects, res
    
@torch.no_grad()
def validation(model, dataset, topk=(1,5), map_output=None, batch_size=250, num_workers=10, shuffle=False, pin_memory=True, mb=None):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(reduction='none')
    filepaths = [(os.path.sep).join(f.split(os.path.sep)[-2:]) for f,_ in dataset.imgs]
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                            shuffle=shuffle, pin_memory=pin_memory)
    
    model.eval()
    results = defaultdict(list)
    count = 0
    for i, batch in enumerate(progress_bar(dataloader, parent=mb)):
        batch_size = batch[0].shape[0]
        images = batch[0].to(device, non_blocking=True)
        target = batch[1].to(device, non_blocking=True)
        index = batch[2].tolist()
        filenames = [filepaths[idx] for idx in index]

        output = model(images)
        if map_output is not None:
            output = map_output(output)
            
        loss = criterion(output, target)

        preds, accuracies, _ = accuracy(output, target, topk=topk)     
        
        # maximum target and non-target activation
        target_activations = torch.gather(output, 1, target.view(-1,1)).squeeze()
        mask = torch.ones_like(output, dtype=bool).scatter_(1, target.view(-1, 1), False)
        non_target_output = output[mask].view(output.shape[0], -1)
        max_nontarget_activation = non_target_output.max(dim=1).values
        snr = target_activations / max_nontarget_activation
        
        results['index'] += index
        results['filenames'] += filenames
        results['target_label'] += target.tolist()
        results['loss'] += loss.tolist()
        results['predicted_label'] += preds[0].tolist()
        results['target_act'] += target_activations.tolist()
        results['max_nontarget_act'] += max_nontarget_activation.tolist()
        results['snr'] += snr.tolist()
        
        for idx,k in enumerate(topk):
            results[f'correct{k}'] += accuracies[idx].tolist()

    df = pd.DataFrame(results)
    return df, None