import torch
from pytorch_metric_learning import losses, distances

from torchvision.transforms import Compose

from datasets.graph import Graph
from models import ResGCN
from transforms.augmentation import RandomSelectSequence, PadSequence, SelectSequenceCenter, \
    PointNoise, RandomFlipLeftRight, RandomMove, RandomFlipSequence, JointNoise, RandomCropSequence, \
    ShuffleSequence, SequentialExtract
from transforms.multi_input import MultiInput

from torch_geometric.data import Data

class ToFlatTensor:
    def __call__(self, data):
        return data.x, data.y, (data.camera_id, data.session_id)


class GaitGraph2Transforms:
    def __init__(self, options):
        sequence_length = options['sequence_length']
        multi_input = options['multi_input']
        multi_branch = options['multi_branch']
        flip_sequence_p = 0.5
        flip_lr_p = 0.5
        joint_noise = 0.1
        point_noise = 0.05
        random_move = (3, 1)
        train_shuffle_sequence = False
        test_shuffle_sequence = False
        confidence_noise = 0.0
        graph_name = options['graph_name']

        self.graph = Graph(graph_name)

        self.transform_train = Compose([
            PadSequence(sequence_length),
            RandomFlipSequence(flip_sequence_p),
            RandomSelectSequence(sequence_length),
            ShuffleSequence(train_shuffle_sequence),
            RandomFlipLeftRight(flip_lr_p, flip_idx=self.graph.flip_idx),
            JointNoise(joint_noise),
            PointNoise(point_noise),
            RandomMove(random_move),
            MultiInput(self.graph.connect_joint, self.graph.center, enabled=multi_input, concat=not multi_branch),
            ToFlatTensor()
        ])
        self.transform_val = Compose([
            PadSequence(sequence_length),
            MultiInput(self.graph.connect_joint, self.graph.center, enabled=multi_input),
        ])

    def transform(self, mode):
        if mode == 'train':
            return self.transform_train
        if mode == 'predict':
            return self.transform_val


class GaitGraph2:
    def __init__(self, options):
        self.options = options

        self.graph = Graph(self.options['graph_name'])
        model_args = {
            "A": torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False),
            "num_class": self.options['embedding_layer_size'],
            "num_input": 3 if self.options['multi_branch'] else 1,
            "num_channel": 5 if self.options['multi_input'] else 3,
            "parts": self.graph.parts,
        }
        if self.options['multi_input'] and not self.options['multi_branch']:
            model_args["num_channel"] = 15

        self.backbone = ResGCN(self.options['backend_name'], **model_args)

        self.distance = distances.LpDistance()
        self.train_loss = losses.SupConLoss(self.options['loss_temperature'], distance=self.distance)

    def fit(self, dataloader):
        '''
        train model with given dataset
        this runs a whole training loop
        '''
        print('starting training loop')

        optimizer = torch.optim.SGD(
            self.backbone.parameters(), 
            lr=self.options['learning_rate']
        )
        lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.options['learning_rate'],
            epochs=self.options['max_epochs'],
            steps_per_epoch=len(dataloader)
        )

        best_loss = 1000000.0
        best_weight = None

        self.backbone.train()
        for epoch in range(self.options['max_epochs']):
            progress_bar = f'[{" " * 40}]'
            print(f'\nepoch {epoch + 1}/{self.options["max_epochs"]}: {progress_bar} loss=NaN         ', end='\r')
            running_loss_total = 0.0
            for batch_index, batch in enumerate(dataloader):
                x, y, _ = batch

                optimizer.zero_grad()

                y_hat = self.backbone(x)[0]

                loss = self.train_loss(y_hat, y.squeeze())
                # self.log("train_loss", loss, on_epoch=True)
                loss.backward()
                optimizer.step()
                lr_schedule.step()

                running_loss_total += loss.item()
                # running_loss = running_loss_total / ((batch_index + 1) * self.options['batch_size'])
                running_loss = running_loss_total / (batch_index + 1)
                progress_fraction = round(batch_index / (len(dataloader) - 1) * 40)
                progress_bar = f'[{"#" * progress_fraction}{" " * (40 - progress_fraction)}]'
                print(f'epoch {epoch + 1}/{self.options["max_epochs"]}: {progress_bar} loss={running_loss:.4f}        ', end='\r')
            if running_loss_total <= best_loss: 
                best_loss = running_loss_total
                best_weight = self.backbone.state_dict()
        
        self.backbone.load_state_dict(best_weight)
        print('\ndone')
        print('stored best weights in model')

    def save_weights(self, file_path: str):
        '''
        save current model weights to file
        '''
        torch.save(self.backbone.state_dict(), file_path)
        print(f'saved weights to {file_path}')

    def load_weights(self, file_path: str):
        '''
        load model weights from file
        '''
        self.backbone.load_state_dict(torch.load(file_path))
        print(f'loaded weights from {file_path}')

    def encode(self, sequence):
        '''
        ouput feature embedding from given pose sequence
        '''
        with torch.no_grad():
            self.backbone.eval()

            bsz = sequence.shape[0]

            multi_input = MultiInput(self.graph.connect_joint, self.graph.center, True)
            x_flipped = torch.stack([
                multi_input(Data(x=d[:, :, 0, :3].flip(0), device=sequence.device)).x for d in sequence
            ])
            x_lr_flipped = torch.stack([
                multi_input(Data(x=d[:, self.graph.flip_idx, 0, :3], device=sequence.device)).x for d in sequence
            ])

            x = torch.cat([sequence, x_flipped, x_lr_flipped], dim=0)

            y_hat = self.backbone(x)[0]

            f1, f2, f3 = torch.split(y_hat, [bsz, bsz, bsz], dim=0)
            y_hat = torch.cat((f1, f2, f3), dim=1)

            return y_hat


if __name__ == "__main__":
    pass