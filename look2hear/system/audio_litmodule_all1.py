import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
#from speechbrain.processing.speech_augmentation import SpeedPerturb
from safetensors.torch import load_file
from look2hear.utils.util_visualize import *
from collections import OrderedDict
from look2hear.models.metricgan import MetricDiscriminator
from look2hear.losses.stft_loss import STFTLoss,STFTLossPenalty,STFTLossPenaltyHard
from look2hear.losses.ssnr_loss import compute_ssnr
def save_spectrogram(mag_tensor: torch.Tensor, save_path: str):
    """
    Save a magnitude spectrogram as an image.

    Args:
        mag_tensor (torch.Tensor): (batch, freq, time) or (freq, time)
        save_path (str): Path to save PNG image.
    """
    # Ensure CPU and detach
    if isinstance(mag_tensor, torch.Tensor):
        mag = mag_tensor.detach().cpu().numpy()
    else:
        mag = mag_tensor

    # If batch dimension exists, take first element
    if mag.ndim == 3:
        mag = mag[0]  # (freq, time)

    # Convert to log scale (optional, improves visibility)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-8))

    # Plot and save
    plt.figure(figsize=(10, 4))
    plt.imshow(mag_db, origin='lower', aspect='auto', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
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
def select_one_sample(mixtures, targets, harmony_score, category):
    # 0~3 중에서 랜덤 선택
    mask = torch.isin(category, torch.tensor([0, 1, 2, 3], device=category.device))
    valid_indices = torch.where(mask)[0]

    if len(valid_indices) == 0:
        return None, None, None, None

    selected_idx = random.choice(valid_indices).item()
    return mixtures[selected_idx], targets[selected_idx], harmony_score[selected_idx], category[selected_idx]

def select_one_from_high_hard(mixtures, targets, harmony_score, category,num=4):
    # 4~6 중 harmony_score 높은 순 3개 중에서 랜덤 선택
    mask = torch.isin(category, torch.tensor([4, 5, 6], device=category.device))
    valid_indices = torch.where(mask)[0]

    if len(valid_indices) == 0:
        return None, None, None, None

    # 해당 인덱스들과 스코어 정렬
    sorted_indices = sorted(valid_indices.tolist(), key=lambda i: harmony_score[i].item(), reverse=True)
    top_indices = sorted_indices[:min(num, len(sorted_indices))]

    selected_idx = random.choice(top_indices)
    return mixtures[selected_idx], targets[selected_idx], harmony_score[selected_idx], category[selected_idx]
def select_one_from_high_hard_unison(mixtures, targets, harmony_score, category,num=3):
    # 4~6 중 harmony_score 높은 순 3개 중에서 랜덤 선택
    mask = torch.isin(category, torch.tensor([0,1,2,3], device=category.device))
    valid_indices = torch.where(mask)[0]
    if len(valid_indices) == 0:
        return None, None, None, None
    # 해당 인덱스들과 스코어 정렬
    sorted_indices = sorted(valid_indices.tolist(), key=lambda i: harmony_score[i].item(), reverse=True)
    #top_indices = sorted_indices[:min(num, len(sorted_indices))]

    selected_idx = random.choice(sorted_indices)
    return mixtures[selected_idx], targets[selected_idx], harmony_score[selected_idx], category[selected_idx]



class AudioLightningModuleAll1(pl.LightningModule):
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
        self.stft_loss=STFTLossPenaltyHard()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        self.default_monitor = "val_loss/dataloader_idx_0"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_step_outputs_refined = []
        self.test_step_outputs2= []
        self.test_step_outputs2_refined = []
        self.win=960
        self.stride=240
        isfirst=True
        self.my_epoch_start=0
        if isfirst:
            resume = torch.load("Experiments/checkpoint/fin_all/20250813_00/epoch=31.ckpt")
            epoch_= resume["epoch"]
            self.my_epoch_start= epoch_ + 230 # Set the current epoch to the loaded epoch
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
                #self.scheduler.load_state_dict(resume["lr_schedulers"][0])
                print(f"🚀🚀🚀 optimizer 로드, shcedulet 로드")
            except Exception as e:
                print(f"🚀🚀🚀 optimizer 로드 실패, shcedulet 로드 실패: {e}")
        
    def get_mag(self, input,compress_factor=0.3):
        stft_spec = torch.stft(input, n_fft=self.win, hop_length=self.stride, 
                          window=torch.hann_window(self.win).to(input.device).type(input.type()),
                          return_complex=True) #(batch,321,401)
        stft_spec = torch.view_as_real(stft_spec)
        mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9)) #(1,321,401)
        mag = torch.pow(mag, compress_factor)
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
        return (g1 + g2) / 2
    def penalty_loss_scheduler(self):
        """Penalty loss scheduler."""
        if self.my_epoch_start+self.current_epoch < 100:
            return 0.0
        else:
            return 0.002
        
    def hard_num_sheduler(self):
        """Hard num scheduler."""
        # if self.my_epoch_start+self.current_epoch < 200:
        #     return 8
        # elif self.my_epoch_start+self.current_epoch < 250:
        #     return 6
        # else:
        #     return 4
        return 8
    def training_step(self, batch, batch_nb):
        mixtures_full, targets_full, harmony_score, category = batch
        num_= self.hard_num_sheduler()
        if random.random() < 0.5:
            mix_sel, target_sel, score_sel, category_sel = select_one_from_high_hard_unison(
                mixtures_full, targets_full, harmony_score, category
            )
            if mix_sel is None:
                print("🚨🚨🚨🚨🚨첫째🚨🚨🚨🚨🚨")
        else:
            mix_sel, target_sel, score_sel, category_sel = select_one_from_high_hard(
                mixtures_full, targets_full, harmony_score, category,num=num_
            )
            if mix_sel is None:
                print("⚠️⚠️⚠️⚠️⚠️둘째⚠️⚠️⚠️⚠️⚠️")
        if mix_sel is None:
            mixtures = mixtures_full[0].unsqueeze(0)
            targets = targets_full[0].unsqueeze(0)
        else:
            mixtures = mix_sel.unsqueeze(0)
            targets = target_sel.unsqueeze(0)

        est_sources,mag_est1,pha_est1,mag_est2,pha_est2 = self(mixtures)
        loss,reorder_sources,batch_indices = self.loss_func["train"](est_sources, targets,return_ests=True)
        mag_target1,pha_target1,_=self.audio_model.mag_pha_stft(targets[:,0])
        mag_target2,pha_target2,_=self.audio_model.mag_pha_stft(targets[:,1])
        mag_target=torch.stack([mag_target1, mag_target2], dim=1) # (batch,2,321,401)
        pha_target=torch.stack([pha_target1, pha_target2], dim=1)
        mag_target_reordered = torch.stack([mag_target[i, perm] for i, perm in enumerate(batch_indices)], dim=0)
        pha_target_reordered = torch.stack([pha_target[i, perm] for i, perm in enumerate(batch_indices)], dim=0)
        mag_loss,pha_loss,penalty_loss=self.stft_loss(mag_target_reordered[:,0],mag_est1,pha_target_reordered[:,0],pha_est1,mag_target_reordered[:,1],mag_est2,pha_target_reordered[:,1],pha_est2) #clean_mag, mag_g, clean_pha, pha_g
        penalty_loss_weight=self.penalty_loss_scheduler()
        
        self.log(
            "train_loss_mag",
            mag_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "train_loss_pha",
            pha_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(   
            "train_penalty_loss",
            penalty_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "num_scheduler",
            num_,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,    
        )
        self.log(
            "penalty_loss_weight_scheduler",
            penalty_loss_weight,
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
            sample_folder="result/unison2/salience_map/"+f"{category[0]}_{loss.item():.3f}_{mag_loss.item():.3f}_{penalty_loss.item():.3f}_{generate_random_string(3)}"
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
                reorder_sources[:, 0, :],
                sample_folder+"/est_sources_wav1.wav",sr=24000
            )
            save_waveform(
                reorder_sources[:, 1, :],
                sample_folder+"/est_sources_wav2.wav",sr=24000
            )
            save_spectrogram(
                mag_target_reordered[:,0],
                sample_folder+"/target_mag1.png"
            )
            save_spectrogram(
                mag_target_reordered[:,1],
                sample_folder+"/target_mag2.png"
            )
            save_spectrogram(
                self.get_mag(mixtures),
                sample_folder+"/mixture_mag.png"
            )
            save_spectrogram(
                mag_est1,
                sample_folder+"/est_sources_mag1.png"
            )
            save_spectrogram(
                mag_est2,
                sample_folder+"/est_sources_mag2.png"
            )
            print(sample_folder+"/est_sources_mag2.png")
        return {"loss": loss + mag_loss*0.1+penalty_loss*penalty_loss_weight}


    def validation_step(self, batch, batch_nb, dataloader_idx):
        # cal val loss
        if dataloader_idx == 0:
            mixtures, targets, _ = batch
            # print(mixtures.shape)
            est_sources,mag_est1,pha_est1,mag_est2,pha_est2= self(mixtures)
            loss,reorder_sources,batch_indices = self.loss_func["val"](est_sources, targets,return_ests=True)
            mag_target1,pha_target1,_=self.audio_model.mag_pha_stft(targets[:,0])
            mag_target2,pha_target2,_=self.audio_model.mag_pha_stft(targets[:,1])
            mag_target=torch.stack([mag_target1, mag_target2], dim=1) # (batch,2,321,401)
            pha_target=torch.stack([pha_target1, pha_target2], dim=1)
            mag_target_reordered = torch.stack([mag_target[i, perm] for i, perm in enumerate(batch_indices)], dim=0)
            pha_target_reordered = torch.stack([pha_target[i, perm] for i, perm in enumerate(batch_indices)], dim=0)
            mag_loss,pha_loss,penalty_loss=self.stft_loss(mag_target_reordered[:,0],mag_est1,pha_target_reordered[:,0],pha_est1,mag_target_reordered[:,1],mag_est2,pha_target_reordered[:,1],pha_est2) #clean_mag, mag_g, clean_pha, pha_g
            ssnr=compute_ssnr(targets[0,0].cpu().numpy(), reorder_sources[0,0].cpu().numpy(),targets[0,1].cpu().numpy(), reorder_sources[0,1].cpu().numpy())
            self.log(
                "val_ssnr",
                ssnr.mean(),
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            self.log(
                "val_loss_mag",
                mag_loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            self.log(
                "val_loss_pha",
                pha_loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            self.log(
                "val_penalty_loss",
                penalty_loss,
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
            
            self.validation_step_outputs.append(loss)
            
            return {"val_loss": loss}

        # cal test loss
        if (self.trainer.current_epoch) % 1 == 0 and dataloader_idx >= 1:
            mixtures, targets, _ = batch
            est_sources,mag_est1,pha_est1,mag_est2,pha_est2 = self(mixtures)
            tloss,reorder_sources,batch_indices = self.loss_func["val"](est_sources, targets,return_ests=True)
            
            mag_target1,pha_target1,_=self.audio_model.mag_pha_stft(targets[:,0])
            mag_target2,pha_target2,_=self.audio_model.mag_pha_stft(targets[:,1])
            mag_target=torch.stack([mag_target1, mag_target2], dim=1) # (batch,2,321,401)
            pha_target=torch.stack([pha_target1, pha_target2], dim=1)
            mag_target_reordered = torch.stack([mag_target[i, perm] for i, perm in enumerate(batch_indices)], dim=0)
            pha_target_reordered = torch.stack([pha_target[i, perm] for i, perm in enumerate(batch_indices)], dim=0)
            mag_loss,pha_loss,penalty_loss=self.stft_loss(mag_target_reordered[:,0],mag_est1,pha_target_reordered[:,0],pha_est1,mag_target_reordered[:,1],mag_est2,pha_target_reordered[:,1],pha_est2) #clean_mag, mag_g, clean_pha, pha_g
            ssnr=compute_ssnr(targets[0,0].cpu().numpy(), reorder_sources[0,0].cpu().numpy(),targets[0,1].cpu().numpy(), reorder_sources[0,1].cpu().numpy())
            self.log(
                f"test_ssnr_{dataloader_idx}",
                ssnr.mean(),
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            
            self.log(
                f"test_penalty_loss_{dataloader_idx}",
                penalty_loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            
            self.log(
                f"test_mag_loss_{dataloader_idx}",
                mag_loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            self.log(
                f"test_phase_loss_{dataloader_idx}",
                pha_loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            self.log(
                f"test_loss_{dataloader_idx}",
                tloss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            if dataloader_idx == 1:
                self.test_step_outputs.append(tloss)
            else:
                self.test_step_outputs2.append(tloss)
            visualize=False
            if visualize:
                sample_folder="result/unison2/salience_map/"+f"test{dataloader_idx}_{tloss.item():.3f}_{mag_loss.item():.3f}_{penalty_loss.item():.3f}_{generate_random_string(3)}"
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
                    reorder_sources[:, 0, :],
                    sample_folder+"/est_sources_wav1.wav",sr=24000
                )
                save_waveform(
                    reorder_sources[:, 1, :],
                    sample_folder+"/est_sources_wav2.wav",sr=24000
                )
                save_spectrogram(
                    self.get_mag(targets[:, 0, :]),
                    sample_folder+"/target_mag1.png"
                )
                save_spectrogram(
                    self.get_mag(targets[:, 1, :]),
                    sample_folder+"/target_mag2.png"
                )
                save_spectrogram(
                    self.get_mag(mixtures),
                    sample_folder+"/mixture_mag.png"
                )
                save_spectrogram(
                    self.get_mag(reorder_sources[:, 0, :]),
                    sample_folder+"/est_sources_mag1.png"
                )
                save_spectrogram(
                    self.get_mag(reorder_sources[:, 1, :]),
                    sample_folder+"/est_sources_mag2.png"
                )
                print(sample_folder+"/est_sources_mag2.png")
            return {"test_loss": tloss}

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
            test_loss = torch.mean(self.all_gather(avg_loss))
            self.logger.experiment.log(
                {"test_pit_sisnr": -test_loss, "epoch": self.current_epoch}
            )
            
            if len(self.test_step_outputs2) > 0:
                avg_loss2 = torch.stack(self.test_step_outputs2).mean()
                test_loss2 = torch.mean(self.all_gather(avg_loss2))
                self.logger.experiment.log(
                    {"test_pit_sisnr2": -test_loss2, "epoch": self.current_epoch}
                )
                
        self.validation_step_outputs.clear()  # free memory
        
        self.test_step_outputs.clear()  # free memory
        self.test_step_outputs2.clear()  # free memory
        
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