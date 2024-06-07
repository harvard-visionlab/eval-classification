from s3_filestore import S3FileStore
from notion_logger import NotionLogger

import pandas as pd
from copy import deepcopy

class ResultsLogger(object):
    def __init__(self, analysis: str, bucket: str, notion_db_name: str, unique_property="hash_id"):
        self.analysis = analysis
        self.bucket = bucket.split("/")[0]
        self.subfolder = "/".join(bucket.split("/")[1:])
        self.s3 = S3FileStore(self.bucket)
        if notion_db_name is not None:
            self.notion = NotionLogger(notion_db_name, unique_property=unique_property)
        else:
            self.notion = None
        
    def _prepare_df(self, df, meta_dict):
        df = df.copy()
        new_columns = list(meta_dict.keys())
        for k,v in meta_dict.items():
            df[k] = v
        remaining_columns = [col for col in df.columns if col not in new_columns]
        new_column_order = new_columns + remaining_columns
        df = df[new_column_order]
        
        return df
    
    def _prepare_summary(self, summary, meta_dict):
        data = {**meta_dict, **summary}
        for k,v in data.items():
            if isinstance(v, dict):
                data[k] = v.__repr__()
                
        return pd.DataFrame(data, index=[0])
    
    def _prepare_meta(self,  meta):
        data = meta.state_dict()
        for k,v in data.items():
            if isinstance(v, dict):
                data[k] = v.__repr__()
                
        return pd.DataFrame(data, index=[0])
    
    def upload(self, results, summary, meta, verbose=False):
        '''save all results as csv files so they can easily be loaded with pandas.read_csv(url)'''
        
        layer_name = results.iloc[0].layer_name
        
        urls = dict()    
        meta_dict = {k:v for k,v in meta.state_dict().items() if k in ['model_name', 'hash_id']}
        results_df = self._prepare_df(results, meta.state_dict())
        object_key = f"{self.subfolder}/{meta.model_name}_{meta.hash_id}_{layer_name}_results.csv"
        _, urls['results_url'] = self.s3.upload_data(results_df, object_key, verbose=verbose)
        
        meta_dict = dict(layer_name=layer_name, **meta_dict)
        summary_df = self._prepare_summary(summary, meta_dict)
        object_key = f"{self.subfolder}/{meta.model_name}_{meta.hash_id}_{layer_name}_summary.csv"
        _, urls['summary_url'] = self.s3.upload_data(summary_df, object_key, verbose=verbose)
                
        meta_df = self._prepare_meta(meta)
        object_key = f"{self.subfolder}/{meta.model_name}_{meta.hash_id}_meta.csv"
        _, urls['meta_url'] = self.s3.upload_data(meta_df, object_key, verbose=verbose)
        
        notion_data = summary_df.iloc[0].to_dict()
        response = self.notion.insert_or_update(notion_data)
        self.notion.append_code_block(response['id'], toggle_text=f"urls_{layer_name}", code_text=urls.__repr__())
        
        return urls, response
    
    def __repr__(self):
        fields = f"analysis={self.analysis!r}, bucket={self.bucket!r}, subfolder={self.subfolder!r}"        
        return f"{self.__class__.__name__}({fields})" + f"\n\n{self.s3.__repr__()}" + f"\n\n{self.notion.__repr__()}"