import fire
import os
import torch
from data import get_dataset, get_loader
import models
import utils as u
from functools import partial
from tqdm import tqdm

def save_features(train_loader, test_loader, model, feature_path, debug):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    features = []
    with torch.no_grad():
        for batch in tqdm(train_loader):
            if isinstance(batch, tuple) or isinstance(batch, list):
                inputs = batch[0]
            else:
                inputs = batch
                if 'labels' in inputs:
                    inputs.pop('labels') # remove for hf datasets
            if hasattr(inputs, 'items'):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            feat = model(inputs).detach().cpu()
            features.append(feat)
            if debug: 
                break
        features = torch.cat(features, dim=0)

    test_features = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if isinstance(batch, tuple) or isinstance(batch, list):
                inputs = batch[0]
            else:
                inputs = batch
                if 'labels' in inputs:
                    inputs.pop('labels') # remove for hf datasets
            if hasattr(inputs, 'items'):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            feat = model(inputs).detach().cpu()
            test_features.append(feat)
            if debug: 
                break
        test_features = torch.cat(test_features, dim=0)
        
    print(f'Features: {features.size()}', f'Test features: {test_features.size()}')
    feature_dict = {'train': features, 'test': test_features}
    if not debug:
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        torch.save(feature_dict, feature_path)
        print(f'Saved features to {feature_path}')

def main(model_class, dataset, batch_size=128, num_workers=0, save_path=None, subset_size=None, debug=False, **kwargs):
    assert save_path is not None, "Please specify a save_path"
    args = locals()
    u.pretty_print_dict(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # out_dim=0 should cause the model to return the features
    model, get_transform, tokenizer, input_collate_fn = models.create_model(model_class, out_dim=0, pretrained=True, extract_features=True, **kwargs)
    
    model.to(device)
    model.eval()

    train_dataset, test_dataset = get_dataset(dataset, get_transform, tokenizer, no_augment=True) # (train, test)
    
    if subset_size is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(subset_size, len(train_dataset))))
        test_dataset = torch.utils.data.Subset(test_dataset, range(min(subset_size, len(test_dataset))))
        print(f"Lowering dataset size: train={len(train_dataset)} -> {subset_size}, test={len(test_dataset)} -> {subset_size}")

    train_loader = get_loader(train_dataset, batch_size, num_workers=num_workers, shuffle=False, input_collate_fn=input_collate_fn)
    test_loader = get_loader(test_dataset, batch_size, num_workers=num_workers, shuffle=False, input_collate_fn=input_collate_fn)
    
    save_features(train_loader, test_loader, model, save_path, debug)
    
if __name__ == '__main__':
    fire.Fire(main)
