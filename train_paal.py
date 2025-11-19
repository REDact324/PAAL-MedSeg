import argparse
import os
import random
import time
import tempfile
import weakref
from typing import List, Sequence

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.trainer import Trainer

from ap_strategy import acc_predictor
from utils import dataset
from BDM_Net import BDM_Net

EXAMPLE_USAGE = """Example\n-------\npython train.py \\\n  --train_data_root /data/labeled \\\n  --initial_list /data/labeled/train.txt \\\n  --val_list /data/labeled/val.txt \\\n  --unlabeled_data_root /data/unlabeled \\\n  --unlabeled_list /data/unlabeled/pool.txt \\\n  --selection_size 16 --selection_patience 3 --max_active_iterations 5\n"""

gpu_list = [0]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)


parser = argparse.ArgumentParser(
    description='BDM-Net with Active Data Selection',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=EXAMPLE_USAGE)
parser.add_argument('--sigma', '-s', type=float, default=5)
parser.add_argument('--train_data_root', type=str, required=True)
parser.add_argument('--val_data_root', type=str, default=None)
parser.add_argument('--unlabeled_data_root', type=str, default=None)
parser.add_argument('--initial_list', type=str, required=True,
                    help='Text/CSV that enumerates the initially labeled training samples.')
parser.add_argument('--val_list', type=str, required=True,
                    help='Validation split file. Will be expanded as more samples are labeled.')
parser.add_argument('--unlabeled_list', type=str, required=True,
                    help='Text/CSV that enumerates the unlabeled pool.')
parser.add_argument('--train_batch_size', type=int, default=40)
parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--selection_batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--img_size', type=int, default=352)
parser.add_argument('--max_epochs_per_cycle', type=int, default=10)
parser.add_argument('--max_active_iterations', type=int, default=5)
parser.add_argument('--selection_patience', type=int, default=3,
                    help='Number of epochs without validation improvement before sampling.')
parser.add_argument('--selection_size', type=int, default=10,
                    help='How many samples to promote from the unlabeled pool per trigger.')
parser.add_argument('--selection_score_type', choices=['mean', 'log_mean'], default='mean')
parser.add_argument('--val_metric', type=str, default='val_mean_dice')
parser.add_argument('--predictor_name', type=str, default='ap18')
parser.add_argument('--predictor_ckpt', type=str, default=None,
                    help='Optional checkpoint for the accuracy predictor.')
parser.add_argument('--load_predictor_ckpt', action='store_true',
                    help='Explicitly load the predictor checkpoint. When omitted the predictor '
                         'will be randomly initialized.')
parser.add_argument('--channels', type=int, default=None,
                    help='Override the number of image channels if auto-detection is incorrect.')
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

def _init_fn(worker_id, seed=42):
    random.seed(seed + worker_id)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def read_data_list_file(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Cannot locate data list: {path}')
    with open(path, 'r') as fp:
        lines = [line.strip() for line in fp if line.strip()]
    return lines


def _write_temp_list(entries: Sequence[str]) -> str:
    fd, tmp_path = tempfile.mkstemp(prefix='al_split_', suffix='.txt')
    with os.fdopen(fd, 'w') as tmp:
        tmp.write('\n'.join(entries))
    return tmp_path


def _safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def build_transforms(img_size):
    train_trfm = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.75, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])
    val_trfm = A.Resize(img_size, img_size)
    return train_trfm, val_trfm


def build_dataset(split: str,
                  data_root: str,
                  data_list,
                  transform,
                  sigma: float):
    cleanup = False
    resolved_list = data_list
    if not isinstance(data_list, str):
        resolved_list = _write_temp_list(data_list)
        cleanup = True
    dataset_obj = dataset.MyDataset(
        split=split,
        data_root=data_root,
        data_list=resolved_list,
        transform=transform,
        sigma=sigma)
    if cleanup:
        weakref.finalize(dataset_obj, _safe_remove, resolved_list)
    return dataset_obj


def model_init(max_epochs: int, num_classes: int):
    model = BDM_Net(nclass=num_classes, max_epoch=max_epochs)
    path = None
    if path:
        pretrained_dict = torch.load(path, map_location='gpu')['state_dict']
        model.load_state_dict(pretrained_dict)
    return model, max_epochs


class ValidationPlateauWatcher(Callback):
    def __init__(self, metric_name: str, patience: int):
        super().__init__()
        self.metric_name = metric_name
        self.patience = patience
        self.best = None
        self.bad_epochs = 0
        self.triggered = False
        self.history: List[float] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        metric = trainer.callback_metrics.get(self.metric_name)
        if metric is None:
            return
        score = metric.item() if isinstance(metric, torch.Tensor) else float(metric)
        self.history.append(score)
        if self.best is None or score > self.best:
            self.best = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.triggered = True
                trainer.should_stop = True

    @property
    def latest_score(self):
        if not self.history:
            return None
        return self.history[-1]


def train_process(model,
                  train_loader,
                  val_loader,
                  max_epochs,
                  val_metric,
                  patience,
                  cycle_idx):
    tb_logger = pl_loggers.TensorBoardLogger('logs/', name='active_cycle', version=cycle_idx)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_dir = os.path.join('logs/active_cycle', 'checkpoints', f'cycle_{cycle_idx}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                          monitor=val_metric,
                                          filename='BDM-{epoch:02d}-{val_mean_dice:.4f}',
                                          save_top_k=5,
                                          mode='max',
                                          save_weights_only=True)
    plateau = ValidationPlateauWatcher(metric_name=val_metric, patience=patience)
    trainer = Trainer(max_epochs=max_epochs,
                      logger=tb_logger,
                      accelerator="auto",
                      precision=16,
                      check_val_every_n_epoch=1,
                      benchmark=True,
                      callbacks=[lr_monitor, checkpoint_callback, plateau])
    trainer.fit(model, train_loader, val_loader)
    return plateau


class SelectionDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        if isinstance(sample, dict):
            image = sample['image']
        elif isinstance(sample, (list, tuple)):
            image = sample[0]
        else:
            image = sample
        return {'image': image}


class ActiveLearningRunner:
    def __init__(self, parsed_args):
        self.args = parsed_args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_root = self.args.train_data_root
        self.val_root = self.args.val_data_root or self.train_root
        self.unlabeled_root = self.args.unlabeled_data_root or self.train_root
        self.train_transform, self.val_transform = build_transforms(self.args.img_size)
        self.labeled_paths = read_data_list_file(self.args.initial_list)
        self.val_paths = read_data_list_file(self.args.val_list)
        self.unlabeled_paths = read_data_list_file(self.args.unlabeled_list)
        self.model, _ = model_init(self.args.max_epochs_per_cycle, self.args.num_classes)
        self.image_channels = self._resolve_image_channels()
        self.predictor = self._build_predictor()
        self.total_selected = 0

    def _resolve_image_channels(self) -> int:
        if self.args.channels is not None:
            return self.args.channels
        if not self.labeled_paths:
            raise RuntimeError('No labeled samples available to infer image channels.')
        probe_dataset = build_dataset(split='val',
                                      data_root=self.train_root,
                                      data_list=self.labeled_paths[:1],
                                      transform=self.val_transform,
                                      sigma=self.args.sigma)
        sample = probe_dataset[0]
        if isinstance(sample, dict):
            image = sample.get('image')
        elif isinstance(sample, (list, tuple)):
            image = sample[0]
        else:
            image = sample
        if image is None:
            raise RuntimeError('Unable to infer image channels from dataset sample.')
        if isinstance(image, torch.Tensor):
            channels = image.shape[0]
        elif isinstance(image, np.ndarray):
            if image.ndim <= 2:
                channels = 1
            else:
                channels = image.shape[0]
        else:
            raise TypeError(f'Unsupported sample type for channel inference: {type(image)}')
        if channels <= 0:
            raise ValueError('Inferred invalid number of channels from dataset sample.')
        print(f'Auto-detected {channels} image channel(s) from the labeled dataset.')
        return int(channels)

    def _build_predictor(self):
        import predictor as predictor_module
        input_channels = self.image_channels + self.args.num_classes
        predictor_fn = getattr(predictor_module, self.args.predictor_name)
        predictor = predictor_fn(num_classes=self.args.num_classes,
                                 input_channels=input_channels)
        if self.args.load_predictor_ckpt:
            if not self.args.predictor_ckpt:
                raise ValueError('load_predictor_ckpt was set but predictor_ckpt is missing.')
            state = torch.load(self.args.predictor_ckpt, map_location='cpu')
            state_dict = state.get('state_dict', state)
            predictor.load_state_dict(state_dict)
        else:
            if self.args.predictor_ckpt:
                print('Predictor checkpoint provided but --load_predictor_ckpt was not set; '
                      'running with randomly initialized predictor instead.')
        predictor = predictor.to(self.device)
        predictor.eval()
        return predictor

    def _build_loader(self, entries: Sequence[str], split: str, batch_size: int, shuffle: bool, drop_last: bool):
        if len(entries) == 0:
            raise RuntimeError(f'No samples available for split {split}.')
        root = self.train_root if split == 'train' else (self.val_root if split == 'val' else self.unlabeled_root)
        transform = self.train_transform if split == 'train' else self.val_transform
        dataset_obj = build_dataset(split=split,
                                    data_root=root,
                                    data_list=entries,
                                    transform=transform,
                                    sigma=self.args.sigma)
        loader = DataLoader(dataset_obj,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=self.args.num_workers,
                            pin_memory=True,
                            worker_init_fn=_init_fn)
        return loader

    def _select_from_unlabeled(self):
        if not self.unlabeled_paths:
            return []
        sample_dataset = build_dataset(split='val',
                                       data_root=self.unlabeled_root,
                                       data_list=self.unlabeled_paths,
                                       transform=self.val_transform,
                                       sigma=self.args.sigma)
        selection_dataset = SelectionDataset(sample_dataset)
        sample_loader = DataLoader(selection_dataset,
                                   batch_size=self.args.selection_batch_size,
                                   shuffle=False,
                                   num_workers=self.args.num_workers,
                                   pin_memory=True,
                                   worker_init_fn=_init_fn)
        sample_num = min(self.args.selection_size, len(self.unlabeled_paths))
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            selected = acc_predictor(seg_net=self.model,
                                     predictor=self.predictor,
                                     unlabeled_data_pool=self.unlabeled_paths,
                                     sample_loader=sample_loader,
                                     sample_nums=sample_num,
                                     sample_weight=None,
                                     al_mode='ap+wps',
                                     score_type=self.args.selection_score_type)
        self.model.train()
        return selected

    def run(self):
        for cycle in range(1, self.args.max_active_iterations + 1):
            print(f'===== Active iteration {cycle} =====')
            train_loader = self._build_loader(self.labeled_paths,
                                              split='train',
                                              batch_size=self.args.train_batch_size,
                                              shuffle=True,
                                              drop_last=len(self.labeled_paths) >= self.args.train_batch_size)
            val_loader = self._build_loader(self.val_paths,
                                            split='val',
                                            batch_size=self.args.val_batch_size,
                                            shuffle=False,
                                            drop_last=False)
            plateau = train_process(model=self.model,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    max_epochs=self.args.max_epochs_per_cycle,
                                    val_metric=self.args.val_metric,
                                    patience=self.args.selection_patience,
                                    cycle_idx=cycle)
            if plateau.triggered and self.unlabeled_paths:
                newly_selected = self._select_from_unlabeled()
                if not newly_selected:
                    print('Sampling was triggered but no data was selected.')
                    break
                self.total_selected += len(newly_selected)
                self.labeled_paths.extend(newly_selected)
                self.val_paths.extend(newly_selected)
                self.unlabeled_paths = [p for p in self.unlabeled_paths if p not in newly_selected]
                print(f'Added {len(newly_selected)} new samples. '
                      f'Remaining unlabeled samples: {len(self.unlabeled_paths)}')
            else:
                print('Validation kept improving or unlabeled pool is empty; stopping active learning loop.')
                break
        print(f'Total number of selected samples: {self.total_selected}')
        return self.total_selected


def main():
    start_time = time.perf_counter()
    seed_everything(seed=args.seed)
    runner = ActiveLearningRunner(args)
    total_selected = runner.run()
    print(f'Number of data selected after training: {total_selected}')
    end_time = time.perf_counter()
    ex_time = (end_time - start_time) / 3600
    print(f"Function execution time: {ex_time} hours")


if __name__ == '__main__':
    main()