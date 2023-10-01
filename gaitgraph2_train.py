from torch.utils.data import DataLoader

from gait_dataset import GaitDataset
from gaitgraph2_model import GaitGraph2, GaitGraph2Transforms


if __name__ == '__main__':
    ''' parameters '''
    dataset_path = './data/csv/biox_dataset_30'
    model_output_path = './best.pth'

    options = {
        'max_epochs': 100, 
        'backend_name': 'resgcn-n21-r8', 
        'learning_rate': 0.005, 
        'loss_temperature': 0.01, 
        'embedding_layer_size': 128, 
        'batch_size': 8, 
        'num_workers': 4, 
        'sequence_length': 60, 
        'multi_input': True, 
        'multi_branch': True, 
        'graph_name': 'mediapipe'
    }

    gg_transform = GaitGraph2Transforms(options)

    train_dataset = GaitDataset(
        dataset_path, 
        transform=gg_transform.transform('train'))
    train_dataloader = DataLoader(
        train_dataset, batch_size=options['batch_size'], 
        num_workers=options['num_workers'], shuffle=True)

    gg_model = GaitGraph2(options)
    
    gg_model.fit(train_dataloader)
    gg_model.save_weights(model_output_path)