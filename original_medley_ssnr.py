import os
import random
import json
from tqdm import tqdm
from pprint import pprint
import look2hear.models
import torchaudio
from torch.utils.data import Dataset
import glob
import look2hear.datas
from look2hear.metrics import MetricsTracker
from look2hear.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn#,save_state_dict_keys
#@from functions import load_ola_func_with_args#, load_w2v_func_with_args, load_w2v_chunk_func_with_args, load_spectral_features_chunk_func_with_args
import soundfile as sf
import librosa
import numpy as np
import torch
import argparse
import pandas as pd
import pyloudnorm as pyln
from asteroid.metrics import get_metrics
#from asteroid.data.librimix_dataset import LibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.utils import tensors_to_device

#from data import MyMedleyVox
#from models import load_model_with_args
from functions import load_ola_func_with_args
from look2hear.utils import str2bool, loudnorm, db2linear
from mir_eval.separation import _bss_decomp_mtifilt, _bss_source_crit
from look2hear.losses.ssnr_loss import compute_ssnr,compute_whole_ssnr,compute_vad_ssnr
from look2hear.utils.vad_combiner import conservative_vad_union_intervals
import yaml
def project(source, target, eps=1e-8):
    # source, target: (B, T)
    scale = torch.sum(source * target, dim=-1, keepdim=True) / (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    return scale * target

def sir_loss(ests, targets, eps=1e-8):
    """
    ests: (B, 2, T) - predicted separated sources
    targets: (B, 2, T) - ground truth sources

    Returns: scalar loss (lower = better separation)
    """
    est0 = ests[:, 0, :]  # (B, T)
    est1 = ests[:, 1, :]
    ref0 = targets[:, 0, :]
    ref1 = targets[:, 1, :]

    # projection of est0 onto ref1 (interference)
    interf0 = project(est0, ref1)  # (B, T)
    target0 = project(est0, ref0)

    # projection of est1 onto ref0 (interference)
    interf1 = project(est1, ref0)
    target1 = project(est1, ref1)

    # SIR = target_energy / interference_energy
    num0 = torch.sum(target0 ** 2, dim=-1)
    denom0 = torch.sum(interf0 ** 2, dim=-1) + eps
    sir0 = 10 * torch.log10(num0 / denom0 + eps)

    num1 = torch.sum(target1 ** 2, dim=-1)
    denom1 = torch.sum(interf1 ** 2, dim=-1) + eps
    sir1 = 10 * torch.log10(num1 / denom1 + eps)

    # maximize SIR → minimize negative
    sir = (sir0 + sir1) / 2
    return -sir.mean()
def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
#COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]
COMPUTE_METRICS = ["si_sdr","sdr"]
def resample_to_24k(waveform, orig_sr, target_sr):
    if waveform.ndim == 1:
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    elif waveform.ndim == 2:
        return np.stack([
            librosa.resample(ch, orig_sr=orig_sr, target_sr=target_sr)
            for ch in waveform
        ])
    else:
        raise ValueError("Unsupported waveform shape")
def _bss_eval_no_permutation(reference_sources, estimated_sources):
    """
    Compute BSS metrics assuming source order is known (no permutation search).
    
    Args:
        reference_sources (np.ndarray): shape (K, T)
        estimated_sources (np.ndarray): shape (K, T) or (K+1, T) with noise ignored
    
    Returns:
        sdr, sir, sar: each of shape (K,)
    """
    K, T = reference_sources.shape
    if estimated_sources.shape[0] > K:
        estimated_sources = estimated_sources[:K]  # Ignore extra channels (e.g., noise)
    
    assert estimated_sources.shape == (K, T), f"Shape mismatch: expected {(K, T)}, got {estimated_sources.shape}"

    sdr = np.empty(K)
    sir = np.empty(K)
    sar = np.empty(K)

    for j in range(K):
        s_true, e_spat, e_interf, e_artif = _bss_decomp_mtifilt(
            reference_sources, estimated_sources[j], j, 512
        )
        sdr[j], sir[j], sar[j] = _bss_source_crit(s_true, e_spat, e_interf, e_artif)

    return sdr, sir, sar

class MyMedleyVox(Dataset):
    """Dataset class for MedleyVox source separation tasks.

    Args:
        task (str): One of ``'unison'``, ``'duet'``, ``'main_vs_rest'`` or
            ``'total'`` :
            * ``'unison'`` for unison vocal separation.
            * ``'duet'`` for duet vocal separation.
            * ``'main_vs_rest'`` for main vs. rest vocal separation (main vs rest).
            * ``'n_singing'`` for N-singing separation. We will use all of the duet, unison, and main vs. rest data.

        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture. Actually, this is fixed to 2 for our tasks. Need to be specified for N-singing training (future work).
        segment (int, optional) : The desired sources and mixtures length in s.
    """

    dataset_name = "MedleyVox"

    def __init__(
        self,
        root_dir='/mnt/bear3/users/jungji/meldey/unison',
        metadata_dir=None,
        task="duet",
        sample_rate=24000,
        n_src=2,
        segment=None,
        return_id=True,
        target_sample_rate=None,  # Default to 24kHz
    ):
        self.root_dir = root_dir  # /path/to/data/test_medleyDB
        self.metadata_dir = "configs/testset_config"  # ./testset/testset_config
        self.task = task.lower()
        self.return_id = return_id
        # Get the csv corresponding to the task
        if self.task == "unison":
            self.total_segments_list = glob.glob(f"{self.root_dir}/unison/**/mixture.wav", recursive=True)
        elif self.task == "duet":
            self.total_segments_list = glob.glob(f"{self.root_dir}/duet/**/mixture.wav", recursive=True)
        elif self.task == "main_vs_rest":
            self.total_segments_list = glob.glob(f"{self.root_dir}/rest/*/*")
        elif self.task == "n_singing":
            self.total_segments_list = glob.glob(f"{self.root_dir}/unison/*/*") + glob.glob(f"{self.root_dir}/duet/*/*") + glob.glob(f"{self.root_dir}/rest/*/*")
        self.segment = segment
        self.sample_rate = sample_rate
        #self.total_segments_list=[x for x in self.total_segments_list if "mixture.wav" in x]
        print(f"sample_rate: {sample_rate}")
        self.n_src = n_src

    def __len__(self):
        return len(self.total_segments_list)

    def __getitem__(self, idx):
        song_name = self.total_segments_list[idx].split("/")[-1]
        segment_name = self.total_segments_list[idx].split("/")[-1]
        mixture_path = self.total_segments_list[idx]
        
        sources_path_list = [mixture_path.replace("mixture.wav", f"s0.wav"), mixture_path.replace("mixture.wav", f"s1.wav")]
        sources_cross_path_list = [mixture_path.replace("mixture.wav", f"s0_estimate.wav"), mixture_path.replace("mixture.wav", f"s1_estimate.wav")]
        sources_list = []
        sources_cross_list = []
        ids = []
        for i, source_path in enumerate(sources_path_list):
            s, sr = torchaudio.load(source_path)
            if sr != self.sample_rate:
                s = torchaudio.functional.resample(s, sr, self.sample_rate)
            sources_list.append(s)
            ids.append(os.path.basename(source_path).replace(".wav", ""))
        for j, source_cross_path in enumerate(sources_cross_path_list):
            s, sr = torchaudio.load(source_cross_path)
            if sr != self.sample_rate:
                s = torchaudio.functional.resample(s, sr, self.sample_rate)
            sources_cross_list.append(s)
        # Read the mixture
        mixture, sr = torchaudio.load(mixture_path)
        if sr != self.sample_rate:
            mixture = torchaudio.functional.resample(mixture, sr, self.sample_rate)
        
        waveform1 = sources_list[0][0]  # mono
        waveform2 = sources_list[1][0]  # mono
        waveform1_cross = sources_cross_list[0][0]  # mono
        waveform2_cross = sources_cross_list[1][0]  # mono
        mixture= mixture[0]  # mono
        # Convert sources to tensor
        sources = torch.stack([waveform1, waveform2])
        sources_cross= torch.stack([waveform1_cross, waveform2_cross])
        json_path= mixture_path.replace("mixture.wav", "metrics.json")
        with open(json_path, "r") as f:
            self.metadata = json.load(f)
            mixture_path=self.metadata['mix_path']
        self.mixture_path = mixture_path
        return mixture, sources, sources_cross, sources_path_list


def main(args):
    compute_metrics = COMPUTE_METRICS

    # Handle device placement
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )#inference_speech_my.py #Experiments/checkpoint/att3/20250731_22/epoch=170.ckpt
    eval_save_dir = args.test_dir="/mnt/bear3/users/jungji/meldey" #Experiments/checkpoint/melt6/epoch=21.ckpt
    args.sample_rate=24000
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    
    test_set = MyMedleyVox(
            root_dir=args.test_dir,
            metadata_dir=args.metadata_dir,
            task=args.singing_task,
            sample_rate=24000,
            n_src=2,
            segment=None,
            return_id=True,
            #target_sample_rate=24000,  # Default to 24kHz
        )  
    
    meter = pyln.Meter(args.sample_rate)
    
    
    ex_save_dir = f"{eval_save_dir}/examples_{args.singing_task}_24k"
    os.makedirs(ex_save_dir, exist_ok=True)
    if args.n_save_ex == -1:
        args.n_save_ex = len(test_set)

    # Randomly choose the indexes of sentences to save.
    save_idx = save_idx = list(range(len(test_set)))

    series_list = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_set))):
            mix, sources, est_sources, sources_path_list = test_set[idx]
            est_sources = est_sources.to(device)
            mix, sources = tensors_to_device([mix, sources], device=device)
            try:
                loss, reordered_sources = loss_func(
                    est_sources[None], sources[None], return_est=True
                )
                mix_np = mix.cpu().data.numpy() #* db2linear(-adjusted_gain, eps=0.0)
                sources_np = sources.cpu().data.numpy() #* db2linear(-adjusted_gain, eps=0.0)
                est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy() #* db2linear(-adjusted_gain, eps=0.0)
                #est_sources_swap = est_sources_np[[1, 0], :]
                utt_metrics = get_metrics(
                    mix_np,
                    sources_np,
                    est_sources_np,
                    sample_rate=args.sample_rate,
                    metrics_list=COMPUTE_METRICS,
                    
                )
                ssnr=compute_ssnr(
                    sources_np[0], est_sources_np[0],
                    sources_np[1], est_sources_np[1],
                )
                ssnr_whole = compute_whole_ssnr(
                    sources_np[0], est_sources_np[0],
                    sources_np[1], est_sources_np[1])
                
                utt_metrics["mix_path"] = test_set.mixture_path
                utt_metrics['ssnr']=ssnr.mean().item()
                utt_metrics['ssnr_whole'] = ssnr_whole[1].mean().item()
                series_list.append(pd.Series(utt_metrics))

                # Save some examples in a folder. Wav files and metrics as text.
                if idx in save_idx:
                    local_save_dir = f"{ex_save_dir}/ex_{idx}/"
                    os.makedirs(local_save_dir, exist_ok=True)
                    sf.write(local_save_dir + "mixture.wav", mix_np, args.sample_rate)
                    # Loop over the sources and estimates
                    for src_idx, src in enumerate(sources_np):
                        sf.write(f"{local_save_dir}/s{src_idx}.wav", src, args.sample_rate)
                    for src_idx, est_src in enumerate(est_sources_np):
                        sf.write(
                            f"{local_save_dir}/s{src_idx}_estimate.wav",
                            est_src,
                            args.sample_rate,
                        )
                    # Write local metrics to the example folder.
                    with open(local_save_dir + "metrics.json", "w") as f:
                        json.dump(utt_metrics, f, indent=0)
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                continue
    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(f"{eval_save_dir}/all_metrics_{args.singing_task}.csv")

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()
    final_results["ssnr"] = all_metrics_df["ssnr"].mean()
    final_results["ssnr_whole"] = all_metrics_df["ssnr_whole"].mean()

    print("Overall metrics :")
    print(f"{args.exp_name}{args.suffix_name}")
    pprint(final_results)

    with open(f"{eval_save_dir}/final_metrics_{args.singing_task}.json", "w") as f:
        json.dump(final_results, f, indent=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model test.py")
    # Added arguments
    parser.add_argument("--target", type=str, default="vocals")
    parser.add_argument(
        "--test_target",
        type=str,
        default="singing",
        choices=["speech", "singing"],
        help="choose",
    )
    parser.add_argument(
        "--singing_task",
        type=str,
        default="unison",
        help="only valid when test_target=='singing'. 'unison' or 'duet' or 'main_vs_rest', or 'n_singing'",
    )
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument(
        "--suffix_name",
        type=str,
        default="",
        help="additional folder name you want to attach on the last folder name of 'exp_name'. for example, '_online'",
    )
    parser.add_argument(
        "--model_dir", type=str, default="/path/to/results/singing_sep"
    )
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument(
        "--use_overlapadd",
        type=str,
        default=None,
        choices=[None, "ola", "ola_norm", "w2v", "w2v_chunk", "sf_chunk"],
        help="use overlapadd functions, ola, ola_norm, w2v will work with ola_window_len, ola_hop_len argugments. w2v_chunk and sf_chunk is chunk-wise processing based on VAD, so you have to specify the vad_method args. If you use sf_chunk (spectral_featrues_chunk), you also need to specify spectral_features.",
    )
    parser.add_argument(
        "--vad_method",
        type=str,
        default="spec",
        choices=["spec", "webrtc"],
        help="what method do you want to use for 'voice activity detection (vad) -- split chunks -- processing. Only valid when 'w2v_chunk' or 'sf_chunk' for args.use_overlapadd.",
    )
    parser.add_argument(
        "--spectral_features",
        type=str,
        default="mfcc",
        choices=["mfcc", "spectral_centroid"],
        help="what spectral feature do you want to use in correlation calc in speaker assignment (only valid when using sf_chunk)",
    )
    parser.add_argument(
        "--w2v_ckpt_dir",
        type=str,
        default="ckpt",
        help="only valid when use_overlapadd is 'w2v or 'w2v_chunk'.",
    )
    parser.add_argument(
        "--w2v_nth_layer_output",
        nargs="+",
        type=int,
        default=[0],
        help="wav2vec nth layer output",
    )
    parser.add_argument(
        "--ola_window_len",
        type=float,
        default=None,
        help="ola window size in [sec]",
    )
    parser.add_argument(
        "--nfft",
        default=512,
        type=int),
    
    parser.add_argument(
        "--ola_hop_len",
        type=float,
        default=None,
        help="ola hop size in [sec]",
    )
    parser.add_argument(
        "--reorder_chunks",
        type=str2bool,
        default=True,
        help="ola reorder chunks",
    )
    parser.add_argument(
        "--use_ema_model",
        type=str2bool,
        default=True,
        help="use ema model or online model? only vaind when args.ema it True (model trained with ema)",
    )

    # Original parameters of test code
    parser.add_argument(
        "--test_dir",
        type=str,
        default="conv_tasnet_24k_convnext_6_3k_from_best_250825_unison",
        help="Test directory including the csv files",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="./testset/testset_config",
        help="Metadata for testset, only for 'main vs. rest' separation",
    )
    parser.add_argument(
        "--speech_task",
        type=str,
        # required=True,
        default="sep_clean",
        help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        # required=True,
        default="eval_results",
        help="Directory in exp_dir where the eval results" " will be stored",
    )
    parser.add_argument(
        "--exp_dir",
        default="/path/to/results/singing_sep",
        help="Experiment root. Evaluation results will saved in '(args.exp_dir)/(args.out_dir)/(args.exp_name'",
    )
    parser.add_argument(
        "--n_save_ex",
        type=int,
        default=10,
        help="Number of audio examples to save, -1 means all",
    )

    parser.add_argument(
        "--save_and_load_eval",
        type=str2bool,
        default=False,
        help="To check the output scale exploding, save and load outputs for eval.",
    )
    parser.add_argument(
        "--save_smaller_output",
        type=str2bool,
        default=False,
        help="To check the output scale exploding, save and load outputs for eval.",
    )

    # Original arguments

    args, _ = parser.parse_known_args()

    main(args)
