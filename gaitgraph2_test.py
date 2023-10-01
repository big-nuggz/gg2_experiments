import numpy as np
import torch
from torch_geometric.data import Data

from gaitgraph2_model import GaitGraph2, GaitGraph2Transforms


if __name__ == '__main__':
    ''' parameters '''
    dataset_path = './data/csv/biox_dataset_30'
    model_weights_path = './best.pth'

    options = {
        'max_epochs': 100, 
        'backend_name': 'resgcn-n21-r8', 
        'learning_rate': 0.005, 
        'loss_temperature': 0.01, 
        'embedding_layer_size': 128, 
        'batch_size': 32, 
        'num_workers': 4, 
        'sequence_length': 60, 
        'multi_input': True, 
        'multi_branch': True, 
        'graph_name': 'mediapipe'
    }

    # prepare test sequences
    files = [
        f'{dataset_path}/cam01/subject01/session01.csv', # gallery
        f'{dataset_path}/cam01/subject01/session02.csv', 
        f'{dataset_path}/cam01/subject02/session02.csv', 
        f'{dataset_path}/cam01/subject03/session02.csv', 
        f'{dataset_path}/cam01/subject04/session02.csv', 
        f'{dataset_path}/cam01/subject05/session02.csv', 
        f'{dataset_path}/cam01/subject06/session02.csv', 
        f'{dataset_path}/cam01/subject07/session02.csv', 
        f'{dataset_path}/cam01/subject08/session02.csv', 
        f'{dataset_path}/cam01/subject09/session02.csv', 
        f'{dataset_path}/cam01/subject10/session02.csv', 
        f'{dataset_path}/cam01/subject11/session02.csv', 
        f'{dataset_path}/cam01/subject12/session02.csv', 
    ]

    subject_names = ['gallery'] + [f'subject{index + 1:02d}' for index in range(len(files) - 1)]

    sequences = [[] for _ in range(len(files))]

    for sequence_index, file in enumerate(files):
        with open(file, 'r', encoding='utf8') as f:
            file_data = f.readlines()[1:]
    
        for row in file_data:
            row = row.split(",")
            frame = row[0]

            keypoints = np.array(row[1:], dtype=np.float32)    

            sequences[sequence_index].append(
                torch.tensor(keypoints.reshape(-1, 3))
            )
        sequences[sequence_index] = torch.stack(sequences[sequence_index])
        
    sequence_start = (len(sequences[0]) - options['sequence_length']) // 2
    sequence_end = sequence_start + options['sequence_length']

    sequences = [Data(x=sequence[sequence_start: sequence_end]) for sequence in sequences]

    # calculate embeddings
    gg_transform = GaitGraph2Transforms(options)
    gg_model = GaitGraph2(options)
    
    gg_model.load_weights(model_weights_path)

    transform = gg_transform.transform('predict')

    embeddings = []
    for sequence in sequences:
        sequence = torch.unsqueeze(transform(sequence).x, 0)
        embedding = gg_model.encode(sequence)
        embeddings.append(embedding)

    # compare distance to first sequence
    print('DISTANCES FROM GALLERY')

    gallery = embeddings[0]
    vector_distances = []
    for embedding in embeddings:
        vector_distances.append(torch.cdist(gallery, embedding)[0, 0])

    for subject_name, distance in zip(subject_names, vector_distances):
        print(f'{subject_name}: {distance.float():.4f}')