from pathlib import Path
from torch.utils.data import DataLoader, random_split, ConcatDataset
from data_loader.task import get_task_labels, get_per_task_classes
from data_loader.dataset import VOCSegmentationIncremental, ADESegmentationIncremental, VOCSegmentationIncrementalMemory, ADESegmentationIncrementalMemory


class VOCIncrementalDataLoader():
    def __init__(self, task, train, val, test, num_workers, pin_memory, memory=None):
        self.task = task
        self.train = train

        self.step = task['step']
        self.name = task['name']
        self.classes_idx_new, self.classes_idx_old = get_task_labels('voc', self.name, self.step)
        self.setting = task['setting']
        self.n_classes = len(list(set(self.classes_idx_new + self.classes_idx_old)))

        self.train_set = VOCSegmentationIncremental(
            setting=self.setting,
            classes_idx_new=self.classes_idx_new,
            classes_idx_old=self.classes_idx_old,
            idxs_path=Path(task['idxs_path']) / "voc" / f"{task['setting']}_{task['name']}_train_{self.step:02d}.npy",
            **train['args'],
        )

        if val['cross_val'] is True:
            train_len = int(0.8 * len(self.train_set))
            val_len = len(self.train_set) - train_len   # select 20% of train dataset
            self.train_set, self.val_set = random_split(self.train_set, [train_len, val_len])
        else:
            # Validatoin using validation set.
            self.val_set = VOCSegmentationIncremental(
                val=True,
                setting=self.setting,
                classes_idx_new=self.classes_idx_new,
                classes_idx_old=self.classes_idx_old,
                idxs_path=Path(task['idxs_path']) / "voc" / f"{task['setting']}_{task['name']}_val_{self.step:02d}.npy",
                **val['args'],
            )
        self.test_set = VOCSegmentationIncremental(
            test=True,
            setting=self.setting,
            classes_idx_new=self.classes_idx_new,
            classes_idx_old=self.classes_idx_old,
            idxs_path=Path(task['idxs_path']) / "voc" / f"{task['setting']}_{task['name']}_test_{self.step:02d}.npy",
            **test['args'],
        )

        self.memory = None
        if self.step > 0 and (memory is not None) and memory['mem_size'] != 0:
            classes_idx_new, classes_idx_old = get_task_labels('voc', self.name, self.step - 1)
            self.prev_train_set = VOCSegmentationIncremental(
                setting=self.setting,
                classes_idx_new=classes_idx_new,
                classes_idx_old=classes_idx_old,
                idxs_path=Path(task['idxs_path']) / "voc" / f"{task['setting']}_{task['name']}_train_{self.step-1:02d}.npy",
                **train['args'],
            )

        self.init_train_kwargs = {'num_workers': num_workers, "pin_memory": pin_memory, "batch_size": train["batch_size"]}
        self.init_val_kwargs = {'num_workers': num_workers, "pin_memory": pin_memory, "batch_size": val["batch_size"]}
        self.init_test_kwargs = {'num_workers': num_workers, "pin_memory": pin_memory, "batch_size": test["batch_size"]}

    def get_memory(self, config, concat=True):
        if self.step > 0:
            self.memory = VOCSegmentationIncrementalMemory(
                setting=self.setting,
                step=self.step,
                classes_idx_new=self.classes_idx_new,
                classes_idx_old=self.classes_idx_old,
                # idxs_path=Path(self.task['idxs_path']) / "voc" / f"{self.task['setting']}_{self.task['name']}_memory.json",
                idxs_path=config.save_dir.parent / f'step_{self.step}' / 'memory.json',
                **self.train['args'],
            )
            if concat is True:
                self.train_set = ConcatDataset([self.train_set, self.memory])

    def get_train_loader(self, sampler=None):
        return DataLoader(self.train_set, **self.init_train_kwargs,
                          drop_last=True, sampler=sampler, shuffle=(sampler is None),)

    def get_val_loader(self, sampler=None):
        return DataLoader(self.val_set, **self.init_val_kwargs,
                          sampler=sampler, shuffle=False,)

    def get_test_loader(self, sampler=None):
        return DataLoader(self.test_set, **self.init_test_kwargs,
                          sampler=sampler, shuffle=False,)
    
    def get_memory_loader(self, sampler=None):
        return DataLoader(self.memory, **self.init_train_kwargs,
                          drop_last=True, sampler=sampler, shuffle=(sampler is None),)

    def get_old_train_loader(self, sampler=None):
        return DataLoader(self.prev_train_set, **self.init_train_kwargs,
                          drop_last=False, sampler=sampler, shuffle=(sampler is None),)

    def __str__(self):
        return f"{self.setting} / {self.name} / step: {self.step}"

    def dataset_info(self):
        if self.memory is not None:
            return f"The number of datasets: {len(self.train_set) - len(self.memory)}+{len(self.memory)} / {len(self.val_set)} / {len(self.test_set)}"
        else:
            return f"The number of datasets: {len(self.train_set)} / {len(self.val_set)} / {len(self.test_set)}"

    def task_info(self):
        return {"setting": self.setting, "name": self.name, "step": self.step,
                "old_class": self.classes_idx_old, "new_class": self.classes_idx_new}

    def get_per_task_classes(self, step=None):
        if step is None:
            step = self.step
        return get_per_task_classes('voc', self.name, step)
    
    def get_task_labels(self, step=None):
        if step is None:
            step = self.step
        return get_task_labels('voc', self.name, step)


class ADEIncrementalDataLoader():
    def __init__(self, task, train, val, test, num_workers, pin_memory, memory=None):
        self.task = task
        self.train = train
        
        self.step = task['step']
        self.name = task['name']
        self.classes_idx_new, self.classes_idx_old = \
            get_task_labels('ade', self.name, self.step)
        self.setting = task['setting']
        self.n_classes = len(list(set(self.classes_idx_new + self.classes_idx_old)))

        self.train_set = ADESegmentationIncremental(
            setting=self.setting,
            classes_idx_new=self.classes_idx_new,
            classes_idx_old=self.classes_idx_old,
            idxs_path=Path(task['idxs_path']) / "ade" / f"{task['setting']}_{task['name']}_train_{self.step:02d}.npy",
            **train['args'],
        )

        if val['cross_val'] is True:
            train_len = int(0.8 * len(self.train_set))
            val_len = len(self.train_set) - train_len   # select 20% of train dataset
            self.train_set, self.val_set = random_split(self.train_set, [train_len, val_len])
        else:
            # Validatoin using validation set.
            self.val_set = ADESegmentationIncremental(
                val=True,
                setting=self.setting,
                classes_idx_new=self.classes_idx_new,
                classes_idx_old=self.classes_idx_old,
                idxs_path=Path(task['idxs_path']) / "ade" / f"{task['setting']}_{task['name']}_val_{self.step:02d}.npy",
                **val['args'],
            )
        self.test_set = ADESegmentationIncremental(
            test=True,
            setting=self.setting,
            classes_idx_new=self.classes_idx_new,
            classes_idx_old=self.classes_idx_old,
            idxs_path=Path(task['idxs_path']) / "ade" / f"{task['setting']}_{task['name']}_test_{self.step:02d}.npy",
            **test['args'],
        )
        
        self.memory = None
        if self.step > 0 and (memory is not None) and (memory['mem_size'] != 0):
            classes_idx_new, classes_idx_old = get_task_labels('ade', self.name, self.step - 1)
            self.prev_train_set = ADESegmentationIncremental(
                setting=self.setting,
                classes_idx_new=classes_idx_new,
                classes_idx_old=classes_idx_old,
                idxs_path=Path(task['idxs_path']) / "ade" / f"{task['setting']}_{task['name']}_train_{self.step-1:02d}.npy",
                **train['args'],
            )

        self.init_train_kwargs = {'num_workers': num_workers, "pin_memory": pin_memory, "batch_size": train["batch_size"]}
        self.init_val_kwargs = {'num_workers': num_workers, "pin_memory": pin_memory, "batch_size": val["batch_size"]}
        self.init_test_kwargs = {'num_workers': num_workers, "pin_memory": pin_memory, "batch_size": test["batch_size"]}

    def get_train_loader(self, sampler=None):
        return DataLoader(self.train_set, **self.init_train_kwargs, drop_last=True,
                          sampler=sampler, shuffle=(sampler is None),)

    def get_val_loader(self, sampler=None):
        return DataLoader(self.val_set, **self.init_val_kwargs,
                          sampler=sampler, shuffle=False,)

    def get_test_loader(self, sampler=None):
        return DataLoader(self.test_set, **self.init_test_kwargs,
                          sampler=sampler, shuffle=False,)

    # Memory
    def get_memory(self, config, concat=True):
        if self.step > 0:
            self.memory = ADESegmentationIncrementalMemory(
                setting=self.setting,
                step=self.step,
                classes_idx_new=self.classes_idx_new,
                classes_idx_old=self.classes_idx_old,
                idxs_path=config.save_dir.parent / f'step_{self.step}' / 'memory.json',
                **self.train['args'],
            )
            if concat is True:
                self.train_set = ConcatDataset([self.train_set, self.memory])

    # Memory
    def get_memory_loader(self, sampler=None):
        return DataLoader(self.memory, **self.init_train_kwargs,
                          drop_last=True, sampler=sampler, shuffle=(sampler is None),)
    
    # Memory
    def get_old_train_loader(self, sampler=None):
        return DataLoader(self.prev_train_set, **self.init_train_kwargs,
                          drop_last=False, sampler=sampler, shuffle=(sampler is None),)

    def __str__(self):
        return f"{self.setting} / {self.name} / step: {self.step}"
    
    def dataset_info(self):
        if self.memory is not None:
            return f"The number of datasets: {len(self.train_set) - len(self.memory)}+{len(self.memory)} / {len(self.val_set)} / {len(self.test_set)}"
        else:
            return f"The number of datasets: {len(self.train_set)} / {len(self.val_set)} / {len(self.test_set)}"

    def task_info(self):
        return {"setting": self.setting, "name": self.name, "step": self.step,
                "old_class": self.classes_idx_old, "new_class": self.classes_idx_new}

    def get_per_task_classes(self, step=None):
        if step is None:
            step = self.step
        return get_per_task_classes('ade', self.name, step)
    
    def get_task_labels(self, step=None):
        if step is None:
            step = self.step
        return get_task_labels('ade', self.name, step)
