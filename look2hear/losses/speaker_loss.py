import torch
import torch.nn as nn
import torch.nn.functional as F
#from singer_identity import load_model
from look2hear.utils.util_visualize import *
import torchaudio
from basic_pitch_torch.model import BasicPitchTorch
from basic_pitch_torch.constants import (
    AUDIO_SAMPLE_RATE,
    AUDIO_N_SAMPLES,
    ANNOTATIONS_FPS,
    FFT_HOP,
)
from basic_pitch_torch.myinference_real import *



class SimpleSpeakerEmbeddingLoss(nn.Module):
    def __init__(self, pitch_scale=2.0):
        super().__init__()
        self.pitch_scale = pitch_scale
        self.speaker_extractor = load_model('byol', input_sr=16000).eval()#.#to(device)
        for param in self.speaker_extractor.parameters():
            param.requires_grad = False
        #self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=44100)#.to(waveform.device)
        self.timbre_trap = BasicPitchTorch()
        self.timbre_trap.load_state_dict(torch.load('assets/basic_pitch_pytorch_icassp_2022.pth'))
        self.timbre_trap.eval()
        print("Timbre Trap loaded")
        self.timbre_trap.eval()
        for param in self.timbre_trap.parameters():
            param.requires_grad = False

    # ========== Embedding Utilities ==========

    def extract_sliding_embeddings(self, waveform, sr_target=16000, win_sec=1, stride_sec=0.5):
        #waveform = self.resampler(waveform)
        device= waveform.device
        total_len = waveform.shape[0]
        win_size = int(win_sec * sr_target)
        stride_size = int(stride_sec * sr_target)
        chunks = []
        visualize=False
        avg_pitches=[]
        for start in range(0, total_len - win_size + 1, stride_size):
            chunk = waveform[start:start + win_size]
            pitch_per_chunk=self.get_pitch_salience(chunk.unsqueeze(0))  # (1,124=time,264=f)
            segment_pitch_bins = torch.softmax(pitch_per_chunk, dim=-1).argmax(dim=-1).squeeze(0) #(124)
            valid_pitch_bins = segment_pitch_bins[(segment_pitch_bins > 50) & (segment_pitch_bins < 250)]
            if valid_pitch_bins.numel() == 0:
                avg_pitch=0.0
            else:
                avg_pitch = valid_pitch_bins.float().mean()#.item()
            avg_pitches.append(avg_pitch)
            if visualize:
                save_path = f"result/sim1/{generate_random_string(3)}_chunk_{start // stride_size}.wav"
                save_waveform(chunk, save_path, sr_target)
            with torch.no_grad():
                emb = self.speaker_extractor(chunk.unsqueeze(0))
            chunks.append(emb.squeeze(0))
        avg_pitches = torch.tensor(avg_pitches, dtype=torch.float32)  # (num_chunks,)
        return torch.stack(chunks).to(device),avg_pitches.to(device)  # (n_chunks, D)

    def extract_embeddings_by_pitch(self, segment_batch_list):
        emb_list = []
        for _seg in segment_batch_list:
            seg=[]
            if len(_seg) == 0:
                emb_list.append(torch.zeros(1, 1000))
            else:
                for s in _seg:
                    #s = self.resampler(s)
                    seg.append(s)  # Add batch dimension
                merged_tensor = torch.cat(seg, dim=0)
                with torch.no_grad():
                    emb = self.speaker_extractor(merged_tensor)
                emb_list.append(emb)
        return torch.stack(emb_list, dim=0)  # (batch, n_chunks, D)

    def extract_spk_emb(self, est_sources):
        emb_list = []
        pitch_list = []
        for est in est_sources:
            emb,pitch = self.extract_sliding_embeddings(est)
            pitch_list.append(pitch)
            emb_list.append(emb)
        return torch.stack(emb_list, dim=0),torch.stack(pitch_list, dim=0) # (batch, n_chunks, D),(batch, num_chunks)

    def compute_per_spk_cosine_sim(self, emb):
        return torch.stack([
            F.cosine_similarity(emb[:, t, :], emb[:, t + 1, :], dim=-1)
            for t in range(emb.size(1) - 1)
        ], dim=1)  # (B, T-1)

    def compute_spk_cosine_sim(self, emb1, emb2):
        B, T, D = emb1.shape
        emb1_norm = F.normalize(emb1, p=2, dim=-1)
        emb2_norm = F.normalize(emb2, p=2, dim=-1)
        sim = torch.matmul(emb1_norm.unsqueeze(2), emb2_norm.unsqueeze(1).transpose(-1, -2)).squeeze(-2)
        return sim  # (B, T, T)

    def compute_soft_pitch_weighted_cosine_sim(self, emb, pitch):
        B, T, D = emb.shape
        emb_norm = F.normalize(emb, dim=-1)
        cosine_sim = torch.sum(emb_norm.unsqueeze(2) * emb_norm.unsqueeze(1), dim=-1)

        pitch_diff = torch.abs(pitch.unsqueeze(2) - pitch.unsqueeze(1))
        soft_weight = torch.exp(-pitch_diff / self.pitch_scale).to(emb.device)

        weighted_sim = cosine_sim * soft_weight
        sum_sim = weighted_sim.sum(dim=(1, 2))
        sum_weight = soft_weight.sum(dim=(1, 2)).clamp(min=1e-6)

        return sum_sim / sum_weight  # (B,)

    def compute_cross_pitch_contrastive_loss(self, emb1, emb2, pitch1, pitch2):
        B, T, D = emb1.shape
        emb1_norm = F.normalize(emb1, dim=-1)
        emb2_norm = F.normalize(emb2, dim=-1)

        cosine_sim = torch.sum(
            emb1_norm.unsqueeze(2) * emb2_norm.unsqueeze(1), dim=-1
        )
        pitch_diff = torch.abs(pitch1.unsqueeze(2) - pitch2.unsqueeze(1))
        soft_weight = torch.exp(-pitch_diff / self.pitch_scale).to(emb1.device)

        weighted_penalty = cosine_sim * soft_weight
        sum_penalty = weighted_penalty.sum(dim=(1, 2))
        sum_weight = soft_weight.sum(dim=(1, 2)).clamp(min=1e-6)

        return (sum_penalty / sum_weight).mean()
    def get_pitch_salience(self,audio_tensor: torch.Tensor) -> torch.Tensor:
        n_overlapping_frames = 30
        overlap_len = n_overlapping_frames * FFT_HOP
        hop_size = AUDIO_N_SAMPLES - overlap_len

        audio_windowed, _, audio_original_length = get_audio_input_torch(audio_tensor, overlap_len, hop_size)
        batch_size = audio_windowed.shape[0]

        audio_windowed = audio_windowed.view(-1, AUDIO_N_SAMPLES).to(audio_tensor.device)  # flatten batches
        output = self.timbre_trap(audio_windowed)["contour"]  # (n_batches, n_times_short, n_freqs)

        unwrapped_output = unwrap_output_torch(output, audio_original_length, n_overlapping_frames, batch_size=batch_size)
        return unwrapped_output
    def get_pitch_per_seg(self,segment_batch):
        pitch_batch = []
        for segment in segment_batch:   
            pitch=self._get_pitch_per_seg(segment)
            pitch_batch.append(pitch)
        return pitch_batch
    def _get_pitch_per_seg(self, segment):
        pitch=[]
        pitch_hat_seg=self.get_pitch_salience(segment) #(batch,249=t,264=f)
        segment_pitch_bins = torch.softmax(_pitch_hat_seg, dim=-1).argmax(dim=-1)
        valid_pitch_bins = segment_pitch_bins[(segment_pitch_bins > 50) & (segment_pitch_bins < 250)]
        if valid_pitch_bins.numel() == 0:
            avg_pitch=0.0
        else:
            avg_pitch = valid_pitch_bins.float().mean().item()
        for j,seg__ in enumerate(segment):
            _pitch_hat_seg=self.get_pitch_salience(seg__) #(batch,249=t,264=f)
            segment_pitch_bins = torch.softmax(_pitch_hat_seg, dim=-1).argmax(dim=-1)
            valid_pitch_bins = segment_pitch_bins[(segment_pitch_bins > 50) & (segment_pitch_bins < 250)]
            if valid_pitch_bins.numel() == 0:
                avg_pitch=0.0
            else:
                avg_pitch = valid_pitch_bins.float().mean().item()
            pitch.append(avg_pitch)
        return pitch
    def forward(self, pitch_hat1, pitch_hat2, est_sources1, est_sources2):
        device = est_sources1.device
        B = est_sources1.shape[0]
        visualize=False
        if visualize:
            save_folder=f"result/sim1/"+generate_random_string(3)
            os.makedirs(save_folder, exist_ok=True)
        try:
            spk_emb1,pitch1 = self.extract_spk_emb(est_sources1)
            spk_emb2,pitch2 = self.extract_spk_emb(est_sources2)
            sim1 = self.compute_per_spk_cosine_sim(spk_emb1).mean()
            sim2 = self.compute_per_spk_cosine_sim(spk_emb2).mean()
            sim3 = self.compute_spk_cosine_sim(spk_emb1, spk_emb2).mean()
            # pitch_sim1 = self.compute_soft_pitch_weighted_cosine_sim(spk_emb1, pitch1).mean()
            # pitch_sim2 = self.compute_soft_pitch_weighted_cosine_sim(spk_emb2, pitch2).mean()
        except Exception as e:
            print(f"Error in speaker embedding extraction: {e}")
            sim1 = torch.tensor(0.0, device=device)
            sim2 = torch.tensor(0.0, device=device)
            sim3 = torch.tensor(0.0, device=device)
            # pitch_sim1 = torch.tensor(0.0, device=device)
            # pitch_sim2 = torch.tensor(0.0, device=device)
        total_loss = (1 - sim1) + (1 - sim2) + sim3
        return total_loss






class SpeakerEmbeddingLoss(nn.Module):
    def __init__(self, pitch_scale=2.0):
        super().__init__()
        self.pitch_scale = pitch_scale
        self.speaker_extractor = load_model('byol', input_sr=16000).eval()#.#to(device)
        for param in self.speaker_extractor.parameters():
            param.requires_grad = False
        #self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=44100)#.to(waveform.device)
        self.timbre_trap = BasicPitchTorch()
        self.timbre_trap.load_state_dict(torch.load('assets/basic_pitch_pytorch_icassp_2022.pth'))
        self.timbre_trap.eval()
        print("Timbre Trap loaded")
        self.timbre_trap.eval()
        for param in self.timbre_trap.parameters():
            param.requires_grad = False

    # ========== Embedding Utilities ==========

    def extract_sliding_embeddings(self, waveform, sr_target=16000, win_sec=2, stride_sec=1.0):
        #waveform = self.resampler(waveform)
        total_len = waveform.shape[0]
        win_size = int(win_sec * sr_target)
        stride_size = int(stride_sec * sr_target)
        chunks = []
        visualize=False
        
        for start in range(0, total_len - win_size + 1, stride_size):
            chunk = waveform[start:start + win_size]
            if visualize:
                save_path = f"result/sim1/{generate_random_string(3)}_chunk_{start // stride_size}.wav"
                save_waveform(chunk, save_path, sr_target)
            with torch.no_grad():
                emb = self.speaker_extractor(chunk.unsqueeze(0))
            chunks.append(emb.squeeze(0))

        return torch.stack(chunks)  # (n_chunks, D)

    def extract_embeddings_by_pitch(self, segment_batch_list):
        emb_list = []
        for _seg in segment_batch_list:
            seg=[]
            if len(_seg) == 0:
                emb_list.append(torch.zeros(1, 1000))
            else:
                for s in _seg:
                    #s = self.resampler(s)
                    seg.append(s)  # Add batch dimension
                merged_tensor = torch.cat(seg, dim=0)
                with torch.no_grad():
                    emb = self.speaker_extractor(merged_tensor)
                emb_list.append(emb)
        return torch.stack(emb_list, dim=0)  # (batch, n_chunks, D)

    def extract_spk_emb(self, est_sources):
        emb_list = []
        for est in est_sources:
            emb = self.extract_sliding_embeddings(est)
            emb_list.append(emb)
        return torch.stack(emb_list, dim=0)  # (batch, n_chunks, D)

    def compute_per_spk_cosine_sim(self, emb):
        return torch.stack([
            F.cosine_similarity(emb[:, t, :], emb[:, t + 1, :], dim=-1)
            for t in range(emb.size(1) - 1)
        ], dim=1)  # (B, T-1)

    def compute_spk_cosine_sim(self, emb1, emb2):
        B, T, D = emb1.shape
        emb1_norm = F.normalize(emb1, p=2, dim=-1)
        emb2_norm = F.normalize(emb2, p=2, dim=-1)
        sim = torch.matmul(emb1_norm.unsqueeze(2), emb2_norm.unsqueeze(1).transpose(-1, -2)).squeeze(-2)
        return sim  # (B, T, T)

    def detect_pitch_change_points(self, pitch_salience_batch, threshold=0.05, default_split=2):
        change_points_list = [
            self._detect_single_pitch_changes(ps, threshold, default_split=default_split)
            for ps in pitch_salience_batch
        ]  # List of 1D LongTensors
        # padding to max_n
        max_len = max(cp.shape[0] for cp in change_points_list)
        padded_change_points = torch.full((len(change_points_list), max_len), fill_value=-1, dtype=torch.long)

        for i, cp in enumerate(change_points_list):
            padded_change_points[i, :cp.shape[0]] = cp

        return padded_change_points  # shape: (batch, max_len)

    def _detect_single_pitch_changes(self, pitch_salience, threshold, default_split=2):
        pitch_soft = torch.softmax(pitch_salience, dim=-1)
        pitch_bins = pitch_soft.argmax(dim=-1)  # (T,)
        diff = torch.abs(pitch_bins[1:] - pitch_bins[:-1])
        change_points = torch.where(diff > threshold * diff.max())[0] + 1  # (N,) or empty

        if change_points.numel() == 0 and pitch_bins.shape[0] >= default_split:
            total_len = pitch_bins.shape[0]
            interval = total_len // default_split
            change_points = torch.arange(interval, total_len, interval, device=pitch_bins.device)

        return change_points  # (change_points_len)


    def extract_pitch_based_segments(self, waveforms, pitch_saliencies, change_points_list, sr=16000, hop_size=256, min_len_sec=2):
        segment_batch = []

        for i in range(pitch_saliencies.shape[0]):
            segments = self._extract_segments_single(
                waveforms[i], pitch_saliencies[i], change_points_list[i], sr, hop_size, min_len_sec
            )
            segment_batch.append(segments) # (n_segments, L)

        # 가장 긴 segment 개수 기준으로 padding
        max_segments = max(segs.shape[0] for segs in segment_batch)
        seg_len = int(min_len_sec * sr)
        B = len(segment_batch)

        padded_segments = torch.zeros((B, max_segments, seg_len))

        for b in range(B):
            seg = segment_batch[b] #(n_segments, L)
            padded_segments[b, :seg.shape[0]] = seg
        return padded_segments #(batch, max_segments, seg_len), segment_batch

    def _extract_segments_single(self, waveform, pitch_salience, change_points, sr=16000, hop_size=256, min_len_sec=2):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        min_len_samples = int(min_len_sec * sr)
        total_len = waveform.shape[1]
        frame_pts = [0] + change_points + [pitch_salience.shape[0]]
        segments = []

        for i in range(len(frame_pts) - 1):
            t_start, t_end = frame_pts[i], frame_pts[i + 1]
            s_start, s_end = t_start * hop_size, t_end * hop_size
            seg_len = s_end - s_start

            if seg_len < min_len_samples // 10:
                #print(f"Skipping very short segment: {seg_len} < {min_len_samples // 10}")
                continue

            if seg_len < min_len_samples:
                needed = min_len_samples - seg_len
                half = needed // 2
                s_start = max(0, s_start - half)
                s_end = min(total_len, s_end + (needed - (s_start - t_start * hop_size)))

            segment = waveform[:, s_start:s_end]
            if segment.shape[-1] > min_len_samples:
                segment = segment[:, :min_len_samples]

            if segment.shape[-1] < min_len_samples:
                #print(f"Skipping short segment after padding: {segment.shape[-1]} < {min_len_samples}")
                continue
            segments.append(segment.squeeze(0))  # (L,)
        return torch.stack(segments) #(chunks_len,32000)

    # ========== Loss Functions ==========

    def compute_soft_pitch_weighted_cosine_sim(self, emb, pitch):
        B, T, D = emb.shape
        emb_norm = F.normalize(emb, dim=-1)
        cosine_sim = torch.sum(emb_norm.unsqueeze(2) * emb_norm.unsqueeze(1), dim=-1)

        pitch_diff = torch.abs(pitch.unsqueeze(2) - pitch.unsqueeze(1))
        soft_weight = torch.exp(-pitch_diff / self.pitch_scale).to(emb.device)

        weighted_sim = cosine_sim * soft_weight
        sum_sim = weighted_sim.sum(dim=(1, 2))
        sum_weight = soft_weight.sum(dim=(1, 2)).clamp(min=1e-6)

        return sum_sim / sum_weight  # (B,)

    def compute_cross_pitch_contrastive_loss(self, emb1, emb2, pitch1, pitch2):
        B, T, D = emb1.shape
        emb1_norm = F.normalize(emb1, dim=-1)
        emb2_norm = F.normalize(emb2, dim=-1)

        cosine_sim = torch.sum(
            emb1_norm.unsqueeze(2) * emb2_norm.unsqueeze(1), dim=-1
        )
        pitch_diff = torch.abs(pitch1.unsqueeze(2) - pitch2.unsqueeze(1))
        soft_weight = torch.exp(-pitch_diff / self.pitch_scale).to(emb1.device)

        weighted_penalty = cosine_sim * soft_weight
        sum_penalty = weighted_penalty.sum(dim=(1, 2))
        sum_weight = soft_weight.sum(dim=(1, 2)).clamp(min=1e-6)

        return (sum_penalty / sum_weight).mean()
    def get_pitch_salience(self,audio_tensor: torch.Tensor) -> torch.Tensor:
        n_overlapping_frames = 30
        overlap_len = n_overlapping_frames * FFT_HOP
        hop_size = AUDIO_N_SAMPLES - overlap_len

        audio_windowed, _, audio_original_length = get_audio_input_torch(audio_tensor, overlap_len, hop_size)
        batch_size = audio_windowed.shape[0]

        audio_windowed = audio_windowed.view(-1, AUDIO_N_SAMPLES).to(audio_tensor.device)  # flatten batches
        output = self.timbre_trap(audio_windowed)["contour"]  # (n_batches, n_times_short, n_freqs)

        unwrapped_output = unwrap_output_torch(output, audio_original_length, n_overlapping_frames, batch_size=batch_size)
        return unwrapped_output
    def get_pitch_per_seg(self,segment_batch):
        pitch_batch = []
        for segment in segment_batch:   
            pitch=self._get_pitch_per_seg(segment)
            pitch_batch.append(pitch)
        return pitch_batch
    def _get_pitch_per_seg(self, segment):
        pitch=[]
        pitch_hat_seg=self.get_pitch_salience(segment) #(batch,249=t,264=f)
        segment_pitch_bins = torch.softmax(_pitch_hat_seg, dim=-1).argmax(dim=-1)
        valid_pitch_bins = segment_pitch_bins[(segment_pitch_bins > 50) & (segment_pitch_bins < 250)]
        if valid_pitch_bins.numel() == 0:
            avg_pitch=0.0
        else:
            avg_pitch = valid_pitch_bins.float().mean().item()
        for j,seg__ in enumerate(segment):
            _pitch_hat_seg=self.get_pitch_salience(seg__) #(batch,249=t,264=f)
            segment_pitch_bins = torch.softmax(_pitch_hat_seg, dim=-1).argmax(dim=-1)
            valid_pitch_bins = segment_pitch_bins[(segment_pitch_bins > 50) & (segment_pitch_bins < 250)]
            if valid_pitch_bins.numel() == 0:
                avg_pitch=0.0
            else:
                avg_pitch = valid_pitch_bins.float().mean().item()
            pitch.append(avg_pitch)
        return pitch
    def forward(self, pitch_hat1, pitch_hat2, est_sources1, est_sources2):
        device = est_sources1.device
        B = est_sources1.shape[0]
        visualize=False
        if visualize:
            save_folder=f"result/sim1/"+generate_random_string(3)
            os.makedirs(save_folder, exist_ok=True)
        # ---------- Pitch similarity ----------
        try:
            pitch_change_list1 = self.detect_pitch_change_points(pitch_hat1, threshold=0.1) #(batch,max_seg)
            pitch_change_list2 = self.detect_pitch_change_points(pitch_hat2, threshold=0.1)

            segs1 = self.extract_pitch_based_segments(est_sources1, pitch_hat1, pitch_change_list1) #(batch, max_segments, seg_len)
            segs2 = self.extract_pitch_based_segments(est_sources2, pitch_hat2, pitch_change_list2)
            pitchs1=self.get_pitch_per_seg(segs1)
            pitchs2=self.get_pitch_per_seg(segs2)
            if visualize:
                for i in range(B):
                    save_path = os.path.join(save_folder, f"pitch_seg1_{i}.png")
                    visualize_pitch_change(pitch_hat1[i], pitch_change_list1[i], save_path)
                    save_segments_as_wav(segs1[i], save_folder)
                    for j,seg__ in enumerate(segs1[i]):
                        _pitch_hat_seg=self.get_pitch_salience(seg__) #(batch,249=t,264=f)
                        segment_pitch_bins = torch.softmax(_pitch_hat_seg, dim=-1).argmax(dim=-1)
                        valid_pitch_bins = segment_pitch_bins[(segment_pitch_bins > 50) & (segment_pitch_bins < 250)]
                        if valid_pitch_bins.numel() == 0:
                            avg_pitch=0.0
                        else:
                            avg_pitch = valid_pitch_bins.float().mean().item()
                        print(f"Segment {j} average pitch: {avg_pitch}")
                        visualize_salience_map_for_basicpitch(_pitch_hat_seg[0][:,50:200].T, f"{save_folder}/pitch_salience1_{j}.png")
                    print(pitchs1)
                    # for j, pitch in enumerate(pitchs1[i]):
                    #     visualize_salience_map_for_basicpitch(pitchs1[i][j], f"{save_folder}/pitch_salience1_{i}_{j}.png")
            emb1 = self.extract_embeddings_by_pitch(segs1)
            emb2 = self.extract_embeddings_by_pitch(segs2)

            pitch1_tensor = torch.tensor(pitchs1, dtype=torch.float32, device=device)
            pitch2_tensor = torch.tensor(pitchs2, dtype=torch.float32, device=device)

            pitch_sim1 = self.compute_soft_pitch_weighted_cosine_sim(emb1, pitch1_tensor).mean()
            pitch_sim2 = self.compute_soft_pitch_weighted_cosine_sim(emb2, pitch2_tensor).mean()
        except Exception as e:
            print(f"Error in pitch extraction: {e}")
            pitch_sim1 = torch.tensor(0.0, device=device)
            pitch_sim2 = torch.tensor(0.0, device=device)

        try:
            spk_emb1 = self.extract_spk_emb(est_sources1)
            spk_emb2 = self.extract_spk_emb(est_sources2)

            sim1 = self.compute_per_spk_cosine_sim(spk_emb1).mean()
            sim2 = self.compute_per_spk_cosine_sim(spk_emb2).mean()
            sim3 = self.compute_spk_cosine_sim(spk_emb1, spk_emb2).mean()
        except Exception as e:
            print(f"Error in speaker embedding extraction: {e}")
            sim1 = torch.tensor(0.0, device=device)
            sim2 = torch.tensor(0.0, device=device)
            sim3 = torch.tensor(0.0, device=device)
        total_loss=0.0#(1 - pitch_sim1) + (1 - pitch_sim2) 
        # ---------- Total loss ----------
        total_loss = + (1 - sim1) + (1 - sim2) + sim3
        return total_loss
