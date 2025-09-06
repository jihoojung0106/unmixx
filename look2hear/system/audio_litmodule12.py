import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
#from speechbrain.processing.speech_augmentation import SpeedPerturb
from safetensors.torch import load_file
from look2hear.utils.util_visualize import *
from collections import OrderedDict
from look2hear.models.metricgan import MetricDiscriminator

def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_

    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def chunk_waveform(waveform, sr=24000, segment_secs=1.5, hop_ratio=0.5):
    """
    Args:
        waveform: Tensor of shape (batch, time)
        sr: Sampling rate (default: 24000)
        segment_secs: Length of each segment in seconds (default: 1.0)
        hop_ratio: Hop size ratio (default: 0.5 means 50%)
    Returns:
        Tensor of shape (batch * n_chunks, segment_len)
    """
    batch_size, total_len = waveform.shape
    segment_len = int(sr * segment_secs)
    hop_len = int(segment_len * hop_ratio)

    chunks = []
    for b in range(batch_size):
        w = waveform[b]
        if total_len < segment_len:
            chunks.append(w)
        else:
            for start in range(0, total_len - segment_len + 1, hop_len):
                chunk = w[start:start+segment_len]
                chunks.append(chunk)
    
    return torch.stack(chunks, dim=0)  # (batch * n_chunks, segment_len)
def has_sound(wav, threshold=1e-2):
    # wav: (batch, time)
    energy = torch.sqrt(torch.mean(wav ** 2, dim=1))  # (batch,)
    return energy > threshold  # (batch,) of bool


class AudioLightningModule12(pl.LightningModule):
    def __init__(
        self,
        audio_model=None,
        video_model=None,
        optimizer=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        # Speed Aug
        # self.speedperturb = SpeedPerturb(
        #     self.config["datamodule"]["data_config"]["sample_rate"],
        #     speeds=[95, 100, 105],
        #     perturb_prob=1.0
        # )
        # Save lightning"s AttributeDict under self.hparams
        #self.discriminator=MetricDiscriminator()
        self.default_monitor = "val_loss/dataloader_idx_0"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        # self.print(self.audio_model)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_step_outputs_refined = []
        self.test_step_outputs2= []
        self.test_step_outputs2_refined = []
        #self.gate=torch.nn.Parameter(torch.tensor(0.5))
        self.win=960
        self.stride=240
        isfirst=True
        if isfirst:
            resume = torch.load("Experiments/checkpoint/fin_van1/20250716_23/epoch=266.ckpt")
            epoch_= resume["epoch"]
            #self.current_epoch = epoch_  # Set the current epoch to the loaded epoch
            #self.trainer.current_epoch = epoch_  # Update the trainer's current epoch
            print(f"🚀🚀🚀 epoch {epoch_} 로드")
            global_step_ = resume["global_step"]
            print(f"🚀🚀🚀 global_step {global_step_} 로드")
            loaded_state_dict= resume["state_dict"]
            model_state_dict = self.audio_model.state_dict()
            filtered_state_dict = OrderedDict()
            unmatched_keys = []  # 여기에 안 맞는 key를 저장
            for k in loaded_state_dict.keys():
                new_key = k.replace("audio_model.", "")
                if new_key in model_state_dict and loaded_state_dict[k].shape == model_state_dict[new_key].shape:
                    filtered_state_dict[new_key] = loaded_state_dict[k]
                new_new_key=k.replace("audio_model.", "").replace("separator.","separator2.") 
                if new_new_key in model_state_dict:
                    filtered_state_dict[new_new_key] = loaded_state_dict[k]
                else:
                    unmatched_keys.append(new_key)  # 로드 안 된 key 기록
            print(f"🚀🚀🚀 총 {len(filtered_state_dict.keys())}개의 키가 로딩되었습니다.")
            if unmatched_keys:
                print(f"⭐️⭐️⭐️ 다음 키는 로드되지 않았습니다: {len(unmatched_keys)}")
            
            self.audio_model.load_state_dict(filtered_state_dict, strict=False)
            try:
                self.optimizer.load_state_dict(resume["optimizer_states"][0])
                self.scheduler.load_state_dict(resume["lr_schedulers"][0])
                print(f"🚀🚀🚀 optimizer 로드, shcedulet 로드")
            except Exception as e:
                print(f"🚀🚀🚀 optimizer 로드 실패, shcedulet 로드 실패: {e}")
    def get_mag(self, input):
        stft_spec = torch.stft(input, n_fft=self.win, hop_length=self.stride, 
                          window=torch.hann_window(self.win).to(input.device).type(input.type()),
                          return_complex=True) #(batch,321,401)
        stft_spec = torch.view_as_real(stft_spec)
        mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9)) #(1,321,401)
        return mag
  
    def forward(self, wav, mouth=None):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.audio_model(wav)
    def sharpened_gate(self,metric_g1, metric_g2, T=1.0):
        g1 = torch.sigmoid(metric_g1 / T)
        g2 = torch.sigmoid(metric_g2 / T)
        
        # g1_unt = torch.sigmoid(metric_g1)
        # g2_unt = torch.sigmoid(metric_g2)
        return (g1 + g2) / 2
    def training_step(self, batch, batch_nb):
        mixtures, targets, category = batch
        
        new_targets = []
        min_len = -1
        if self.config["training"]["SpeedAug"] == True:
            with torch.no_grad():
                for i in range(targets.shape[1]):
                    new_target = self.speedperturb(targets[:, i, :])
                    new_targets.append(new_target)
                    if i == 0:
                        min_len = new_target.shape[-1]
                    else:
                        if new_target.shape[-1] < min_len:
                            min_len = new_target.shape[-1]

                targets = torch.zeros(
                            targets.shape[0],
                            targets.shape[1],
                            min_len,
                            device=targets.device,
                            dtype=torch.float,
                        )
                for i, new_target in enumerate(new_targets):
                    targets[:, i, :] = new_targets[i][:, 0:min_len]
                    
                mixtures = targets.sum(1)
        # print(mixtures.shape)
        est_sources,est_sources_final = self(mixtures)
        loss,reorder_sources,batch_indices = self.loss_func["train"](est_sources, targets,return_ests=True)
        target_wav_reordered = torch.stack([targets[i, perm] for i, perm in enumerate(batch_indices)], dim=0)
        loss_refined= self.loss_func["train"].no_pit_forward(est_sources_final, target_wav_reordered)
        
        self.log(
            "train_loss_refined",
            loss_refined,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        visualize=False
        if visualize:
            sample_folder="result/base1/salience_map/"+f"{category[0]}_{loss.item():.3f}_{generate_random_string(5)}"
            os.makedirs(sample_folder, exist_ok=True)
            save_waveform(
                targets[:, 0, :],
                sample_folder+"/target_wav1.wav",sr=24000
            )
            save_waveform(
                targets[:, 1, :],
                sample_folder+"/target_wav2.wav",sr=24000
            )
            save_waveform(
                mixtures,
                sample_folder+"/mixture.wav",sr=24000
                
            )
            save_waveform(
                est_sources[:, 0, :],
                sample_folder+"/est_sources_wav1.wav",sr=24000
            )
            save_waveform(
                est_sources[:, 1, :],
                sample_folder+"/est_sources_wav2.wav",sr=24000
            )
        return {"loss": loss+loss_refined}


    def validation_step(self, batch, batch_nb, dataloader_idx):
        # cal val loss
        if dataloader_idx == 0:
            mixtures, targets, _ = batch
            # print(mixtures.shape)
            est_sources,est_sources_final = self(mixtures)
            
            loss,reorder_sources,batch_indices = self.loss_func["val"](est_sources, targets,return_ests=True)
            target_wav_reordered = torch.stack([targets[i, perm] for i, perm in enumerate(batch_indices)], dim=0)
            loss_refined= self.loss_func["val"].no_pit_forward(est_sources_final, target_wav_reordered)
            
            
            
            
            
            self.log(
                "val_loss",
                loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            self.log(
                "val_loss_refined",
                loss_refined,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            
            self.validation_step_outputs.append(loss_refined)
            
            return {"val_loss": loss_refined}

        # cal test loss
        if (self.trainer.current_epoch) % 1 == 0 and dataloader_idx >= 1:
            mixtures, targets, _ = batch
            est_sources,est_sources_final = self(mixtures)
            
            tloss,reorder_sources,batch_indices = self.loss_func["val"](est_sources, targets,return_ests=True)
            target_wav_reordered = torch.stack([targets[i, perm] for i, perm in enumerate(batch_indices)], dim=0)
            tloss_refined= self.loss_func["val"].no_pit_forward(est_sources_final, target_wav_reordered)
            self.log(
                f"test_loss_{dataloader_idx}",
                tloss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            self.log(
                f"test_loss_refined_{dataloader_idx}",
                tloss_refined,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            if dataloader_idx == 1:
                self.test_step_outputs.append(tloss)
                self.test_step_outputs_refined.append(tloss_refined)
            else:
                self.test_step_outputs2.append(tloss)
                self.test_step_outputs2_refined.append(tloss_refined)
            return {"test_loss": tloss_refined}

    def on_validation_epoch_end(self):
        # val
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.logger.experiment.log(
            {"learning_rate": self.optimizer.param_groups[0]["lr"], "epoch": self.current_epoch}
        )
        self.logger.experiment.log(
            {"val_pit_sisnr": -val_loss, "epoch": self.current_epoch}
        )

        # test
        if (self.trainer.current_epoch) % 1 == 0:
            avg_loss = torch.stack(self.test_step_outputs).mean()
            avg_loss_refined = torch.stack(self.test_step_outputs_refined).mean()
            test_loss = torch.mean(self.all_gather(avg_loss))
            test_loss_refined = torch.mean(self.all_gather(avg_loss_refined))
            self.logger.experiment.log(
                {"test_pit_sisnr": -test_loss, "epoch": self.current_epoch}
            )
            self.logger.experiment.log(
                {"test_pit_sisnr_refined": -test_loss_refined, "epoch": self.current_epoch}
            )
            if len(self.test_step_outputs2) > 0:
                avg_loss2 = torch.stack(self.test_step_outputs2).mean()
                avg_loss2_refined = torch.stack(self.test_step_outputs2_refined).mean()
                test_loss2 = torch.mean(self.all_gather(avg_loss2))
                test_loss2_refined = torch.mean(self.all_gather(avg_loss2_refined))
                self.logger.experiment.log(
                    {"test_pit_sisnr2": -test_loss2, "epoch": self.current_epoch}
                )
                self.logger.experiment.log(
                    {"test_pit_sisnr2_refined": -test_loss2_refined, "epoch": self.current_epoch}
                )
        self.validation_step_outputs.clear()  # free memory
        
        self.test_step_outputs.clear()  # free memory
        self.test_step_outputs2.clear()  # free memory
        self.test_step_outputs_refined.clear()  # free memory
        self.test_step_outputs2_refined.clear()  # free memory
    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     if metric is None:
    #         scheduler.step()
    #     else:
    #         scheduler.step(metric)
    
    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return [self.val_loader, self.test_loader]

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic