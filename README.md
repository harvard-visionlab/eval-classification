# eval-classification
Python code for running classification evals on DNN models


### general use

All that's required is a model, and a dataset that returns imgs,labels with the appropriate image transforms.

```
from torchvision import models, datasets
from torchvision.models import AlexNet_Weights
from eval_classification import validate

model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
transform = AlexNet_Weights.IMAGENET1K_V1.transforms()
dataset = datasets.ImageFolder("/path/to/ImageNet2012/val", transform=transform)
results, summary = validate(model, dataset)
summary
```

### visionlab use

For visionlab use, you can use access lab-standardized datasets and models, and automatically upload results to an s3_bucket and/or a notion database. Access to private lab data requires aws access keys / notion integration tokens.

```
from visionlab_datasets import load_dataset
from visionlab_models import load_model
from eval_classification import Classifier, ResultsLogger

# setup classifier with automatic remote data storage
classifier = Classifier(bucket='visionlab-results/alvarez/Projects/dnn-evals/eval-classification', 
                        notion_db='')

model, transforms = load_model("torchvision/alexnet", weights="DEFAULT")
dataset, output_mapping = load_dataset("in1k", split="val", transform=transforms['val'])

results, summary, urls = classifier.validate(model, dataset, output_mapping)
```

### You can also run a linear probe or knn evaluation on the outputs of any named module

If you are training linear probes, it's recommended that you run this on the cluster, rather than your local workstation.

```
from visionlab_datasets import load_dataset, load_dataloader
from visionlab_models import load_model
from eval_classification import Classifier, ResultsLogger

# setup classifier with automatic remote data storage
classifier = Classifier(bucket='visionlab-results/alvarez/Projects/dnn-evals/eval-classification', 
                        notion_db='')

model, transforms = load_model("torchvision/alexnet", weights="DEFAULT")
train_loader, train_output_mapping = load_dataloader("in1k", split="train", transform=transforms['train'])
val_loader, val_output_mapping = load_dataloader("in1k", split="val", transform=transforms['val'])

layer_names = ['classifier.1', 'classifier.3', 'classifier.5']

probe_results = classifier.linear_probe(model, train_loader, val_loader, layer_names, epochs=10,
                                        train_output_mapping=train_output_mapping, val_output_mapping=val_output_mapping)

knn_results = classifier.knn(model, train_loader, val_loader, layer_names, K=200,
                             train_output_mapping=train_output_mapping, val_output_mapping=val_output_mapping)

```
