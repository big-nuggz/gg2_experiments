'''
DATASET STRUCTURE:
dataset_root/
  config.txt
  cam01/
    subject01/
      session01.csv

config.txt contains information about dataset
there must be 3 numbers separated by new lines
for example, if there are 2 cameras, 10 subjects, and 5 sessions, 
the contents of the file will be as following (it must be in order):
2
10
5
'''
import os
from typing import Optional, Callable
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, Data


class GaitDataset(InMemoryDataset):
    def __init__(
            self, dataset_path: str, 
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None):
        '''
        dataset used for training the GaitGraph2 model
        '''
        self.dataset_path = dataset_path
        self._load_data_paths()

        self.split = 'train'
        super().__init__(self.dataset_path, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return f"gait_dataset.pt"

    def _load_data_paths(self) -> None:
        self.cameras, self.subjects, self.sessions = read_config(f'{self.dataset_path}/config.txt')

        self.data_paths = []
        for subject in range(1, self.subjects + 1):
            for session in range(1, self.sessions + 1):
                for camera in range(1, self.cameras + 1):
                    formatted_text = self._format_csv_path(camera, subject, session)
                    if os.path.exists(formatted_text): 
                        self.data_paths.append(((subject, session, camera), formatted_text))
    
    def _format_csv_path(self, camera: int, subject: int, session: int) -> str:
        '''
        path formatter
        re-implement this according to dataset structure
        '''
        return f'{self.dataset_path}/cam{camera:02d}/subject{subject:02d}/session{session:02d}.csv'
    
    def process(self):
        sequences = {}

        for data_path in self.data_paths:
            key, path = data_path
            with open(path) as f:
                samples = f.readlines()[1:]

            sequences[key] = []

            for row in samples:
                row = row.split(",")
                frame = row[0]

                keypoints = np.array(row[1:], dtype=np.float32)    

                sequences[key].append(
                    torch.tensor(keypoints.reshape(-1, 3))
                )

        data_list = []
        for key, keypoints in tqdm(sequences.items(), f"process [{self.split}]"):
            (camera_id, subject_id, session_id) = key

            if len(keypoints) == 0:
                continue

            data = Data(
                x=torch.stack(keypoints),
                y=int(subject_id),
                camera_id=int(camera_id),
                session_id=int(session_id),
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def read_config(path: str) -> tuple:
    '''
    config.txt reading helper function
    '''
    data = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.rstrip()
            if line[0] == '#':
                continue
            data.append(int(line))
    return data[0], data[1], data[2]


# test
if __name__ == '__main__':
    dataset_path = './data/csv/biox_dataset_30'
    dataset = GaitDataset(dataset_path)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4)

    for data in dataset:
        print(data.x)
        break