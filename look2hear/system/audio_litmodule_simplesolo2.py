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


class AudioLightningModuleSimpleSolo2(pl.LightningModule):
    def __init__(
        self,
        audio_model=None,discriminator=None,
        video_model=None,
        optimizer=None,optimizer_d=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        scheduler=None,scheduler_d=None,
        config=None,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.optimizer_d= optimizer_d
        self.automatic_optimization=False
        self.scheduler_d = scheduler_d
        self.discriminator = discriminator
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        #self.discriminator=MetricDiscriminator()
        self.default_monitor = "val_loss/dataloader_idx_0"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        # self.print(self.audio_model)
        self.validation_step_outputs = []
        self.validation_step_outputs_d=[]
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
            
            missing,unmatched=self.audio_model.load_state_dict(filtered_state_dict, strict=False)
            print(f"🚀🚀🚀 Missing keys: {len(missing)}, Unmatched keys: {len(unmatched)}")
            try:
                self.optimizer.load_state_dict(resume["optimizer_states"][0])
                self.scheduler.load_state_dict(resume["lr_schedulers"][0])
                print(f"🚀🚀🚀 optimizer 로드, shcedulet 로드")
            except Exception as e:
                print(f"🚀🚀🚀 optimizer 로드 실패, shcedulet 로드 실패: {e}")
            # state_dict = torch.load("Experiments/checkpoint/gan_dynamic5/20250720_11/epoch=3.ckpt")["state_dict"]
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     if k.startswith("discriminator."):
            #         new_key = k[len("discriminator."):]  # prefix 제거
            #         new_state_dict[new_key] = v
            # missing,unload=discriminator.load_state_dict(new_state_dict,strict=False)
            # print(f"🚀🚀🚀 discriminator Missing keys: {len(missing)}, Unloaded keys: {len(unload)}")
        self.bce_loss= torch.nn.BCEWithLogitsLoss()
        self.threshold=-7
    def get_mag(self, input):
        stft_spec = torch.stft(input, n_fft=self.win, hop_length=self.stride, 
                          window=torch.hann_window(self.win).to(input.device).type(input.type()),
                          return_complex=True) #(batch,321,401)
        stft_spec = torch.view_as_real(stft_spec)
        mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9)) #(1,321,401)
        return mag
    def sharpened_gate(self,metric_g1, metric_g2, T=1.0):
        g1 = torch.sigmoid(metric_g1 / T)
        g2 = torch.sigmoid(metric_g2 / T)
        return (g1 + g2) / 2
  
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
    def prepare_discriminator_loss(self,mixtures, targets, reorder_est):
        chunked_mixtures = chunk_waveform(mixtures) #(4,36000)
        chunked_targets1 = chunk_waveform(targets[:,0])
        chunked_targets2 = chunk_waveform(targets[:,1])
        chunked_targets= torch.stack((chunked_targets1, chunked_targets2), dim=1)  # (b*chunk,2,16000)
        chunked_est_sources_reorder_1 = chunk_waveform(reorder_est[:,0])
        chunked_est_sources_reorder_2 = chunk_waveform(reorder_est[:,1])
        chunked_est_sources_reorder = torch.stack((chunked_est_sources_reorder_1, chunked_est_sources_reorder_2), dim=1)  # (b*chunk,2,16000)
        before_mean_loss_chunked = self.loss_func["train"](chunked_est_sources_reorder, chunked_targets,before_mean=True)
        both_have_sound = has_sound(chunked_targets1) & has_sound(chunked_targets2)
        mixture_labels = (~both_have_sound).float()
        valid_mask = (before_mean_loss_chunked > self.threshold) & (mixture_labels != 1.0)
        valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)  # shape: (N_valid,)
        
        mix_mag=self.get_mag(chunked_mixtures) #(b*chunk,321,401)
        target_mag1=self.get_mag(chunked_targets1)
        target_mag2=self.get_mag(chunked_targets2)
        est_mag1=self.get_mag(chunked_est_sources_reorder_1)
        est_mag2=self.get_mag(chunked_est_sources_reorder_2)
        return {
            "valid_indices": valid_indices,
            "mixture_labels": mixture_labels,
            "mix_mag": mix_mag,
            "target_mag1": target_mag1,
            "target_mag2": target_mag2,
            "est_mag1": est_mag1,
            "est_mag2": est_mag2,
            'valid_mask': valid_mask
        }
    def training_step(self, batch, batch_nb):
        mixtures, targets, category = batch
        optimizer,optimizer_d = self.optimizers()
        
        est_sources,est_sources_final = self(mixtures)
        loss,reorder_est,batch_indices = self.loss_func["train"](est_sources, targets,return_ests=True)
        
        chunked_mixtures = chunk_waveform(mixtures) #(4,36000)
        chunk_batch_size = chunked_mixtures.shape[0]
        one_labels = torch.ones(chunk_batch_size).to(mixtures.device, non_blocking=True)
        prepared=self.prepare_discriminator_loss(mixtures, targets, est_sources)
        mix_mag=prepared["mix_mag"]
        target_mag1=prepared["target_mag1"]
        target_mag2=prepared["target_mag2"]
        est_mag1=prepared["est_mag1"]
        est_mag2=prepared["est_mag2"]
        valid_indices = prepared["valid_indices"]
        mixture_labels = prepared["mixture_labels"]
        valid_mask= prepared["valid_mask"]
        
        optimizer_d.zero_grad()  
        metric_m= self.discriminator(mix_mag) #mix
        metric_s1 = self.discriminator(target_mag1) #separated
        metric_s2 = self.discriminator(target_mag2) #separated
        metric_g1 = self.discriminator(est_mag1.detach()) #
        metric_g2 = self.discriminator(est_mag2.detach()) #
        loss_disc_m = self.bce_loss(metric_m.flatten(),mixture_labels) #이건 
        loss_disc_s = self.bce_loss(metric_s1.flatten(),one_labels) + self.bce_loss(metric_s2.flatten(),one_labels) #이건 무조건 1이고 
        if len(valid_indices) > 0:
            est_mag1_valid = est_mag1[valid_indices]
            est_mag2_valid = est_mag2[valid_indices]
            zero_labels_valid = torch.zeros(len(valid_indices), device=mixtures.device)
            metric_g1_valid = self.discriminator(est_mag1_valid.detach())
            metric_g2_valid = self.discriminator(est_mag2_valid.detach())
            loss_disc_g = self.bce_loss(metric_g1_valid.flatten(), zero_labels_valid) + \
                self.bce_loss(metric_g2_valid.flatten(), zero_labels_valid)
            valid_count = valid_mask.sum().item()  # 정수형 개수
            valid_count = max(valid_count * 2, 1)
            loss_disc_all = loss_disc_m + (loss_disc_s) + loss_disc_g/(valid_count*2)
        else:
            loss_disc_all = loss_disc_m + loss_disc_s
        self.manual_backward(loss_disc_all)
        optimizer_d.step()
        
        
        
        optimizer.zero_grad()
        chunked_est_sources_final1 = chunk_waveform(est_sources_final[:,0])
        chunked_est_sources_final2 = chunk_waveform(est_sources_final[:,1])
        est_mag_final1=self.get_mag(chunked_est_sources_final1)
        est_mag_final2=self.get_mag(chunked_est_sources_final2)
        metric_g1 = self.discriminator(est_mag_final1) #
        metric_g2 = self.discriminator(est_mag_final2) #
        loss_metric = (self.bce_loss(metric_g1.flatten(), one_labels)+
                       self.bce_loss(metric_g2.flatten(), one_labels))/2
        target_wav_reordered = torch.stack([targets[i, perm] for i, perm in enumerate(batch_indices)], dim=0)
        loss_refined= self.loss_func["train"].no_pit_forward(est_sources_final, target_wav_reordered)
        loss_gen_all= loss + loss_metric + loss_refined #*2
        self.manual_backward(loss_gen_all)
        self.clip_gradients(optimizer, gradient_clip_val=5.0, gradient_clip_algorithm="norm")
        optimizer.step()
        
        self.log(
            "train_loss_metric",
            loss_metric,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
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
        self.log(   
            "train_loss_disc",
            loss_disc_all,
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
        return {"loss": loss+loss_refined+loss_metric}


    def validation_step(self, batch, batch_nb, dataloader_idx):
        # cal val loss
        if dataloader_idx == 0:
            mixtures, targets, _ = batch
            # print(mixtures.shape)
            est_sources,est_sources_final = self(mixtures)
            loss,reorder_est,batch_indices = self.loss_func["val"](est_sources, targets,return_ests=True)
            
            chunked_mixtures = chunk_waveform(mixtures) #(4,36000)
            chunk_batch_size = chunked_mixtures.shape[0]
            one_labels = torch.ones(chunk_batch_size).to(mixtures.device, non_blocking=True)
            
            prepared=self.prepare_discriminator_loss(mixtures, targets, est_sources)
            mix_mag=prepared["mix_mag"]
            target_mag1=prepared["target_mag1"]
            target_mag2=prepared["target_mag2"]
            est_mag1=prepared["est_mag1"]
            est_mag2=prepared["est_mag2"]
            valid_indices = prepared["valid_indices"]
            mixture_labels = prepared["mixture_labels"]
            valid_mask= prepared["valid_mask"]
            
            
            metric_m= self.discriminator(mix_mag) #mix
            metric_s1 = self.discriminator(target_mag1) #separated
            metric_s2 = self.discriminator(target_mag2) #separated
            metric_g1 = self.discriminator(est_mag1.detach()) #
            metric_g2 = self.discriminator(est_mag2.detach()) #
            loss_disc_m = self.bce_loss(metric_m.flatten(),mixture_labels) #이건 
            loss_disc_s = self.bce_loss(metric_s1.flatten(),one_labels) + self.bce_loss(metric_s2.flatten(),one_labels) #이건 무조건 1이고 
            if len(valid_indices) > 0:
                est_mag1_valid = est_mag1[valid_indices]
                est_mag2_valid = est_mag2[valid_indices]
                zero_labels_valid = torch.zeros(len(valid_indices), device=mixtures.device)
                metric_g1_valid = self.discriminator(est_mag1_valid.detach())
                metric_g2_valid = self.discriminator(est_mag2_valid.detach())
                loss_disc_g = self.bce_loss(metric_g1_valid.flatten(), zero_labels_valid) + \
                    self.bce_loss(metric_g2_valid.flatten(), zero_labels_valid)
                valid_count = valid_mask.sum().item()  # 정수형 개수
                valid_count = max(valid_count * 2, 1)
                loss_disc_all = loss_disc_m + (loss_disc_s) + loss_disc_g/(valid_count*2)
            else:
                loss_disc_all = loss_disc_m + loss_disc_s
            
            
            chunked_est_sources_final1 = chunk_waveform(est_sources_final[:,0])
            chunked_est_sources_final2 = chunk_waveform(est_sources_final[:,1])
            est_mag_final1=self.get_mag(chunked_est_sources_final1)
            est_mag_final2=self.get_mag(chunked_est_sources_final2)
            metric_g1 = self.discriminator(est_mag_final1) #
            metric_g2 = self.discriminator(est_mag_final2) #
            loss_metric = (self.bce_loss(metric_g1.flatten(), one_labels)+
                        self.bce_loss(metric_g2.flatten(), one_labels))/2
            target_wav_reordered = torch.stack([targets[i, perm] for i, perm in enumerate(batch_indices)], dim=0)
            loss_refined= self.loss_func["val"].no_pit_forward(est_sources_final, target_wav_reordered)
            
            self.log(
                "val_loss_metric",
                loss_metric,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
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
            self.log(
                "val_loss_disc",
                loss_disc_all,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            
            self.validation_step_outputs.append(loss_refined)
            self.validation_step_outputs_d.append(loss_disc_all)
            return {"val_loss": loss_refined}

        # cal test loss
        if (self.trainer.current_epoch) % 1 == 0 and dataloader_idx >= 1:
            mixtures, targets, _ = batch
            est_sources,est_sources_final = self(mixtures)
            tloss,reorder_est,batch_indices = self.loss_func["val"](est_sources, targets,return_ests=True)
            
            chunked_mixtures = chunk_waveform(mixtures) #(4,36000)
            chunk_batch_size = chunked_mixtures.shape[0]
            one_labels = torch.ones(chunk_batch_size).to(mixtures.device, non_blocking=True)
            chunked_est_sources_final1 = chunk_waveform(est_sources_final[:,0])
            chunked_est_sources_final2 = chunk_waveform(est_sources_final[:,1])
            est_mag_final1=self.get_mag(chunked_est_sources_final1)
            est_mag_final2=self.get_mag(chunked_est_sources_final2)
            metric_g1 = self.discriminator(est_mag_final1) #
            metric_g2 = self.discriminator(est_mag_final2) #
            loss_metric = (self.bce_loss(metric_g1.flatten(), one_labels)+
                        self.bce_loss(metric_g2.flatten(), one_labels))/2
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
        avg_loss_d = torch.stack(self.validation_step_outputs_d).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))
        val_loss_d = torch.mean(self.all_gather(avg_loss_d))
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        if isinstance(self.scheduler_d, ReduceLROnPlateau):
            self.scheduler_d.step(val_loss_d)
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "disc_lr",
            self.optimizer_d.param_groups[0]["lr"],
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
        self.validation_step_outputs_d.clear()  # free memory
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
        if not isinstance(self.scheduler_d, (list, tuple)):
            self.scheduler_d = [self.scheduler_d]
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
        epoch_schedulers_d = []
        for sched in self.scheduler_d:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers_d.append(sched)
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
                epoch_schedulers_d.append(sched)
        return [self.optimizer,self.optimizer_d], epoch_schedulers+epoch_schedulers_d

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