import torch
import torch.nn as nn

from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config, logger, gpu):
        self.config = config
        
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['epochs'] if cfg_trainer['save_period'] == -1 else cfg_trainer['save_period']
        self.validation_period = cfg_trainer['validation_period'] if cfg_trainer['validation_period'] == -1 else cfg_trainer['validation_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.reset_best_mnt = cfg_trainer['reset_best_mnt']
        self.rank = torch.distributed.get_rank()

        if logger is None:
            self.logger = config.get_logger('trainer', cfg_trainer['verbosity'])
        else:
            self.logger = logger
            # setup visualization writer instance
            if self.rank == 0:
                self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
            else:
                self.writer = TensorboardWriter(config.log_dir, self.logger, False)
        
        if gpu is None:
            # setup GPU device if available, move model into configured device
            self.device, self.device_ids = self._prepare_device(config['n_gpu'])
        else:
            self.device = gpu
            self.device_ids = None

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # if config.resume is not None:
        #     self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result, val_flag = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            
            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.rank == 0:
                if val_flag and (self.mnt_mode != 'off'):
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        self.logger.warning("Warning: Metric '{}' is not found. "
                                            "Model performance monitoring is disabled.".format(self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        self._save_best_model(epoch)
                        
                    else:
                        not_improved_count += 1

                    if (self.early_stop > 0) and (not_improved_count > self.early_stop):
                        self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                         "Training stops.".format(self.early_stop))
                        break

                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch)

        # close TensorboardX
        self.writer.close()
    
    def test(self):
        result = self._test()
        
        if self.rank == 0:
            log = {}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

    def progress(self, logger, i, total_length):
        period = total_length // 5
        if period == 0:
            return
        elif (i % period == 0):
            logger.info(f'[{i}/{total_length}]')

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        arch = type(self.model).__name__
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
            }
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
            }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _save_best_model(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        arch = type(self.model).__name__
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
                # 'config': self.config
            }
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
                # 'config': self.config
            }
        best_path = str(self.checkpoint_dir / 'model_best.pth')
        torch.save(state, best_path)
        self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path, test=False):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location='cpu')
        self.start_epoch = checkpoint['epoch'] + 1
        if not self.reset_best_mnt:
            self.mnt_best = checkpoint['monitor_best']

        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])

        if test is False:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
