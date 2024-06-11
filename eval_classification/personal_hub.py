import os
import pandas as pd
import json
import torch
from torchvision import transforms
from pathlib import Path

from visionlab_datasets import load_dataset
from visionlab_models.hashid import get_hash_id_from_weights_url
import s3_filestore.filestore as filestore

from .version import __version__
from .main import validation

from pdb import set_trace

class ClassifierStore(object):
    """
        ClassifierStore is a class designed for visionlab members to run classification on deep learning models and manage a results cache using amazon s3 buckets. 

        This class automates the process of storing evaluation results locally and in the your S3 bucket, making it easy to 
        retrieve results for previously evaluated models even across different projects.
        
        Usage assumes you have access to these s3 buckets: visionlab-datasets (read) and visionlab-results/username (write)
                
    Attributes:
        dataset (str): Name of the dataset used for evaluation.
        split (str): The data split used (e.g., 'train', 'val').
        results_dir (str): Directory where local results are stored.        
        bucket_name (str): Name of the S3 bucket.
        hub_root (str): optional, override default root directory in the user's S3 bucket for storing results.
        profile (str): AWS profile name.
        endpoint_url (str): S3 endpoint URL.
        acl (str): Access control list for S3 objects.
        hash_length (int): Length of the hash used for object keys.
        expires_in_seconds (int): Expiry time for pre-signed URLs.
        username (str): The S3 username, derived from environment variable `LFS_AWS_USER`.
        s3 (S3FileStore): Instance of the S3FileStore for managing S3 interactions.
        __version__ (str): Version of the ClassifierStore class.

    Methods:
        __init__: Initializes the ClassifierStore with the provided configuration.
    """
    def __init__(self, dataset, split, results_dir, hub_root=None, bucket_name='visionlab-results', profile='wasabi', 
                 endpoint_url='https://s3.wasabisys.com', acl='public-read', hash_length=10,
                 expires_in_seconds=3600):
        self.eval_name = f'eval-classification'
        self.dataset = dataset
        self.split = split
        self.results_dir = os.path.join(results_dir, 'dnn-evals')
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        self.username = os.environ.get('LFS_AWS_USER')
        self.hub_root = f"{self.username}/dnn-evals/" if hub_root is None else hub_root
        self.hub_root = self.hub_root.strip("/")
        assert self.username is not None, f"s3_username cannot be None"
        self.s3 = filestore.S3FileStore(bucket_name, profile=profile, endpoint_url=endpoint_url,
                                        acl=acl, hash_length=hash_length, expires_in_seconds=expires_in_seconds)
        self.__version__ = __version__
        
    def get_meta(self, dataset, weights_file_or_url):
        '''
            Get the relevant metadata for this eval, which is used
            to uniquely identify the model (hash_id from weights_file_or_url)
            and the dataset (name and id), and key parameters known to 
            affect validation accuracy (resize, interp, crop size).            
        '''
        assert hasattr(dataset,'name') and isinstance(dataset.name, str), "Dataset must have a name. Set dataset.name" 
        assert hasattr(dataset,'id') and isinstance(dataset.name, str), "Dataset must have an id. Set dataset.id" 
        meta = dict()
        meta['model_id'] = get_hash_id_from_weights_url(weights_file_or_url)
        meta['dataset_name'] = dataset.name
        meta['dataset_id'] = dataset.id
        meta['eval_id'] = self.__version__
        for t in dataset.transform.transforms:
            if isinstance(t, transforms.Resize):
                meta['resize'] = t.size
                meta['interp'] = t.interpolation.value
            elif isinstance(t, transforms.CenterCrop):
                assert t.size[0]==t.size[1], f"Oops, expected crop dimensions to be equal, got {t.size}"
                meta['crop'] = t.size[0]
        
        return meta
    
    @classmethod
    def format_filenames(cls, meta, layer_name):
        # get results filename
        rawdata_template = "{model_id}_{layer_name}_resize{resize}_crop{crop}_{interp}_rawdata.csv"
        rawdata_filename = rawdata_template.format(**dict(**meta, layer_name=layer_name))
        summary_template = "{model_id}_{layer_name}_resize{resize}_crop{crop}_{interp}_summary.csv"
        summary_filename = summary_template.format(**dict(**meta, layer_name=layer_name))
        
        return dict(rawdata=rawdata_filename, summary=summary_filename)
    
    def get_bucket_prefix(self, dataset_id):
        prefix = f"{self.hub_root}/{self.eval_name}/{self.__version__}/{dataset_id}"
        return prefix
    
    def get_analysis_dir(self, dataset_id):
        analysis_dir = f"/{self.eval_name}/{dataset_id}"
        return analysis_dir.strip("/")
    
    def format_object_keys(self, filenames):
        keys = dict()
        for name,filename in filenames.items():
            keys[name] = f"{self.hub_root}/{self.eval_name}/{self.__version__}/{filename}"
        return keys
            
    def get_dataset(self, transform):
        dataset = load_dataset(self.dataset, split=self.split, fmt="images", transform=transform)
        
        return dataset
    
    def load_local_copies(self, filenames, analysis_dir):
        results = {}
        for name,filename in filenames.items():
            fullfile = os.path.join(self.results_dir, analysis_dir, filename)
            if os.path.isfile(fullfile):
                results[name] = self.s3.load_file(fullfile)
            else:
                results[name] = None
        
        return results
    
    def save_local_copies(self, results, filenames, analysis_dir):
        target_dir = os.path.join(self.results_dir, analysis_dir)
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        for name,filename in filenames.items():
            fullfile = os.path.join(target_dir, filename)
            data = results[name]
            if fullfile.endswith(".csv") and isinstance(data, pd.DataFrame):
                data.to_csv(fullfile, index=False)
            elif fullfile.endswith(".json"):
                with open(fullfile, 'w') as f:
                    json.dump(data, f)
            elif fullfile.endswith(".pth") or fullfile.endswith(".pt") or fullfile.endswith(".pth.tar"):
                torch.save(data, fullfile)

        return results
    
    def save_remote_copies(self, results, keys, verbose=True):
        for name,data in results.items():
            self.s3.upload_data(data, keys[name], verbose=verbose)
            
    def bucket_has_all_files(self, filenames):
        keys = self.format_object_keys(filenames)
        for name,key in keys.items():
            if self.s3.file_exists(key)==False:
                return False
        return True
    
    def bucket_has_all_keys(self, keys):        
        for name,key in keys.items():
            if self.s3.file_exists(key)==False:
                return False
        return True
    
    def any_missing_data(self, data_dict):
        return any([v is None for v in data_dict.values()])
    
    def __call__(self, model, transform, weights_file_or_url, layer_names=None, recompute=False,
                 batch_size=250, num_workers=None, default_output_name='output', mb=None):
        
        num_workers = max(10, len(os.sched_getaffinity(0))) if num_workers is None else num_workers
        dataset = self.get_dataset(transform)
        meta = self.get_meta(dataset, weights_file_or_url)
        analysis_dir = self.get_analysis_dir(dataset.id)
        
        check_layer_names = [default_output_name] if layer_names is None else layer_names
        run_layer_names = []
        results = dict()
        if recompute == True:
            run_layer_names = layer_names
        else:
            for layer_name in check_layer_names:
                filenames = self.format_filenames(meta, layer_name)
                keys = self.format_object_keys(filenames)
                layer_results = self.load_local_copies(filenames, analysis_dir)

                if self.any_missing_data(layer_results) and self.bucket_has_all_keys(keys):
                    for name,key in keys.items():
                        print(f"==> File not found locally, attempting to fetch object {key}")
                        layer_results[name] = self.s3.load_object(key)

                    if not self.any_missing_data(layer_results):
                        print(f"==> {layer_name} files found in bucket, saving local copies:")
                        self.save_local_copies(layer_results, filenames, analysis_dir)

                elif not self.any_missing_data(layer_results):
                    print(f"==> Local copies found layer {layer_name!r}: {filenames}")

                if self.any_missing_data(layer_results):
                    run_layer_names.append(layer_name)

                results[layer_name] = layer_results
        
        # run the validation
        if run_layer_names:
            print(f"==> Running validation for layer_names={run_layer_names}")
            # "layer_names" are hooked, but when None will be default_output_name
            if run_layer_names == [default_output_name] and layer_names is None:
                run_layer_names = None
            layer_results = validation(model, dataset, layer_names=run_layer_names, topk=(1,5), 
                                       batch_size=batch_size, num_workers=num_workers, 
                                       shuffle=False, pin_memory=True, 
                                       default_output_name=default_output_name, 
                                       meta=meta,
                                       mb=None)
            
            for layer_name, results_df in layer_results.items(): 
                results[layer_name] = results_df
                filenames = self.format_filenames(meta, layer_name)
                keys = self.format_object_keys(filenames)
                self.save_local_copies(results[layer_name], filenames, analysis_dir)
                self.save_remote_copies(results[layer_name], keys)
                
        for layer_name,layer_results in results.items():
            assert self.any_missing_data(layer_results)==False, f"Failed to get results for layer_name={layer_name}"
        
        print(f"==> Obtained validation scores for all layers: {list(results.keys())}")
        return results
    
    def __repr__(self):
        rep = (
            f"{self.__class__.__name__}("
            f"dataset={self.dataset!r}, "
            f"split={self.split!r}, "
            f"username={self.username!r}, "
            f"hub_root={self.hub_root!r}, "
            f"results_dir={self.results_dir!r}"
        )
        rep += ")"
        
        return rep