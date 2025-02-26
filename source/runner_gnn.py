import os

import torch

from source.model.gnn import DragGNN_XL
from source.train import load_and_test_model, train_and_evaluate, setup_seed, \
    config, get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")
print(device)


def model_train_and_evaluate():
    setup_seed(config['seed'])
    # Prepare data
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config['dataset_path'],
                                                                        config['aero_coeff'],
                                                                        config['subset_dir'], config['batch_size'])

    # Initialize model
    model = DragGNN_XL().to(device)
    # model = initialize_model(config)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    # Train and evaluate
    train_and_evaluate(model, train_dataloader, val_dataloader, config)

    # Load and test both the best and final models
    final_model_path = os.path.join('models', f'{config["exp_name"]}_final_model.pth')
    print("Testing the final model:")
    load_and_test_model(final_model_path, test_dataloader, device)

    best_model_path = os.path.join('models', f'{config["exp_name"]}_best_model.pth')
    print("Testing the best model:")
    load_and_test_model(best_model_path, test_dataloader, device)


if __name__ == '__main__':
    model_train_and_evaluate()
