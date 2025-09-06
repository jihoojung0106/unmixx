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
from torch.nn.functional import sigmoid
#from data import MyMedleyVox
#from models import load_model_with_args
#from functions import load_ola_func_with_args
from look2hear.utils import str2bool, loudnorm, db2linear
from look2hear.models.metricgan import MetricDiscriminator
import os
import torch
import torchaudio
from collections import OrderedDict
from torch.nn.functional import sigmoid

import yaml
def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
#COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]
COMPUTE_METRICS = ["si_sdr", "sdr"]
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
        root_dir,
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
            self.total_segments_list = glob.glob(f"{self.root_dir}/unison/*/*")
        elif self.task == "duet":
            self.total_segments_list = glob.glob(f"{self.root_dir}/duet/*/*")
        elif self.task == "main_vs_rest":
            self.total_segments_list = glob.glob(f"{self.root_dir}/rest/*/*")
        elif self.task == "n_singing":
            self.total_segments_list = glob.glob(f"{self.root_dir}/unison/*/*") + glob.glob(f"{self.root_dir}/duet/*/*") + glob.glob(f"{self.root_dir}/rest/*/*")
        self.segment = segment
        self.sample_rate = sample_rate
        print(f"sample_rate: {sample_rate}")
        self.n_src = n_src

    def __len__(self):
        return len(self.total_segments_list)

    def __getitem__(self, idx):
        song_name = self.total_segments_list[idx].split("/")[-2]
        segment_name = self.total_segments_list[idx].split("/")[-1]
        mixture_path = (
            f"{self.total_segments_list[idx]}/mix/{song_name} - {segment_name}.wav"
        )
        self.mixture_path = mixture_path
        sources_path_list = glob.glob(f"{self.total_segments_list[idx]}/gt/*.wav")

        if self.task == "main_vs_rest" or self.task == "n_singing":
            if os.path.exists(
                f"{self.metadata_dir}/V1_rest_vocals_only_config/{song_name}.json"
            ):
                metadata_json_path = (
                    f"{self.metadata_dir}/V1_rest_vocals_only_config/{song_name}.json"
                )
            elif os.path.exists(
                f"{self.metadata_dir}/V2_vocals_only_config/{song_name}.json"
            ):
                metadata_json_path = (
                    f"{self.metadata_dir}/V2_vocals_only_config/{song_name}.json"
                )
            else:
                print("main vs. rest metadata not found.")
                raise AttributeError
            with open(metadata_json_path, "r") as json_file:
                metadata_json = json.load(json_file)
        if self.task != "main_vs_rest" and self.task != "n_singing":
            # Read sources
            sources_list = []
            ids = []
            for i, source_path in enumerate(sources_path_list):
                s, sr = torchaudio.load(source_path)
                if sr != self.sample_rate:
                    s = torchaudio.functional.resample(s, sr, self.sample_rate)
                sources_list.append(s)
                ids.append(os.path.basename(source_path).replace(".wav", ""))
            # Read the mixture
            mixture_un, sr = torchaudio.load(mixture_path)
            if sr != self.sample_rate:
                mixture = torchaudio.functional.resample(mixture_un, sr, self.sample_rate)
            
            waveform1 = sources_list[0][0]  # mono
            waveform2 = sources_list[1][0]  # mono
            mixture= mixture[0]  # mono
            # Convert sources to tensor
            sources = torch.stack([waveform1, waveform2])
            if not self.return_id:
                return mixture, sources
            # 5400-34479-0005_4973-24515-0007.wav
            # id1, id2 = mixture_path.split("/")[-1].split(".")[0].split("_")
        else:
            gt_main_name = metadata_json[segment_name]["main_vocal"]
            gt_main, sr = torchaudio.load(
                f"{self.total_segments_list[idx]}/gt/{gt_main_name} - {segment_name}.wav")
            if sr != self.sample_rate:
                gt_main = torchaudio.functional.resample(gt_main, sr, self.sample_rate)
            gt_main=gt_main[0]
            gt_rest_list = metadata_json[segment_name]["other_vocals"]
            ids= [f"{gt_main_name} - {segment_name}"]
            
            #ids.append(f"{gt_main_name} - {segment_name}")
            rest_sources_list = []
            for other_vocal_name in gt_rest_list:
                s, sr = torchaudio.load(
                    f"{self.total_segments_list[idx]}/gt/{other_vocal_name} - {segment_name}.wav")
                if sr != self.sample_rate:
                    s = torchaudio.functional.resample(s, sr, self.sample_rate)
                rest_sources_list.append(s[0])
                #ids.append(f"{other_vocal_name} - {segment_name}")
            gt_rest = torch.stack(rest_sources_list, dim=0).sum(dim=0)
            # sources_list.append(gt_main)
            # sources_list.append(gt_rest)
            mixture, sr = torchaudio.load(mixture_path)
            if sr != self.sample_rate:
                mixture = torchaudio.functional.resample(mixture, sr, self.sample_rate)
            mixture = mixture[0]
            sources = torch.stack([gt_main, gt_rest])
        return mixture, sources, ids
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


def get_mag(input):
        stft_spec = torch.stft(input, n_fft=960, hop_length=240, 
                          window=torch.hann_window(960).to(input.device).type(input.type()),
                          return_complex=True) #(batch,321,401)
        stft_spec = torch.view_as_real(stft_spec)
        mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9)) #(1,321,401)
        return mag
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator=MetricDiscriminator()
    discriminator.eval()
    state_dict = torch.load("Experiments/checkpoint/gan_dynamic5/20250720_11/epoch=3.ckpt")["state_dict"]

    # prefix 제거
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("discriminator."):
            new_key = k[len("discriminator."):]  # prefix 제거
            new_state_dict[new_key] = v
    missing,unload=discriminator.load_state_dict(new_state_dict,strict=False)
    print(f"Missing keys: {len(missing)}, Unloaded keys: {len(unload)}")
    discriminator.to(device)
    compute_metrics = COMPUTE_METRICS

    # Handle device placement
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )#inference_speech_my.py
    model_path="Experiments/checkpoint/fin_van1/20250716_23/epoch=266.ckpt" #Experiments/checkpoint/ver1/20240621_07/config.yaml
    exp_dir = os.path.dirname(model_path) #Experiments/checkpoint/melt6/epoch=21.ckpt
    conf_dir = os.path.join(exp_dir,"conf.yml") #Experiments/checkpoint/ver1/epoch=109.ckpt
    #Experiments/checkpoint/combine2/20250621_10/epoch=120.ckpt
    if args.use_overlapadd:
        eval_save_dir = (
                f"{exp_dir}/result_{args.use_overlapadd}"
            )
    else:
        eval_save_dir = (
            f"{exp_dir}/result_{os.path.basename(model_path).split('.')[0]}"
        )
    with open(conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    config={}
    config["train_conf"] = train_conf
    
    model_class = getattr(look2hear.models, config["train_conf"]["audionet"]["audionet_name"])
    model = model_class(
        sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
        **config["train_conf"]["audionet"]["audionet_config"],
    )
    try:
        state_dict = torch.load(model_path, map_location='cpu')["state_dict"]  # or 'cuda'
        model.load_state_dict(state_dict)
        print("한 번에 로드 성공")  
    except :
        state_dict = torch.load(model_path, map_location='cpu')["state_dict"]
        # audio_model. prefix 제거
        converted_state_dict = {}
        prefix = "audio_model."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]  # prefix 제거
                converted_state_dict[new_key] = v
        # 모델에 로드 (strict=False는 일부 키 mismatch 허용)
        model.load_state_dict(converted_state_dict, strict=True)
        print(set(converted_state_dict.keys()).difference(set(model.state_dict().keys())))
        print("한 번에 로드 실패, prefix 제거 후 로드 성공")    
    model.eval()
    
    args.sample_rate=24000
   # args.seq_dur=4
    # Used to reorder sources only
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
    if args.use_overlapadd:  # Default to ola if not specified
        continuous_nnet = load_ola_func_with_args(args, model, device, meter).to(device)
    else:
        model.to(device)
    
    # Define overlap add functions
    

    ex_save_dir = f"{eval_save_dir}/examples_{args.singing_task}_24k"
    os.makedirs(ex_save_dir, exist_ok=True)
    if args.n_save_ex == -1:
        args.n_save_ex = len(test_set)

    # Randomly choose the indexes of sentences to save.
    save_idx = save_idx = list(range(len(test_set)))

    series_list = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_set))):
            # Forward the network on the mixture.
            mix, sources, ids = test_set[idx]

            # Apply loudness normalization
            #mix, adjusted_gain = loudnorm(mix.numpy(), -24.0, meter, eps=0.0)
            #mix = torch.as_tensor(mix, dtype=torch.float32)
            #sources = sources.numpy() * db2linear(adjusted_gain, eps=0.0)
            #sources = torch.as_tensor(sources, dtype=torch.float32)
            mix, sources = tensors_to_device([mix, sources], device=device)
            try:
                if args.use_overlapadd:
                    if mix.shape[-1]<24000*4: #
                        est_sources = model(mix.unsqueeze(0),istest=True)
                    else:
                        est_sources = continuous_nnet(mix.unsqueeze(0).unsqueeze(0))
                else:
                    est_sources = model(mix.unsqueeze(0),istest=True)
                if isinstance(est_sources, dict):
                    est_sources = est_sources["output_final"]#["second_est_speech"]
                elif isinstance(est_sources, tuple):
                    if "split2" in conf_dir or "time" in conf_dir or "combine1" in conf_dir:
                        est_sources = est_sources[1]
                    elif len(est_sources) == 2:
                        est_sources = est_sources[1]
                    elif len(est_sources) == 3 or len(est_sources) == 5:
                        est_sources = est_sources[0]
                #est_sources=resampler(est_sources)
                
                loss, reordered_sources = loss_func(
                    est_sources, sources[None], return_est=True
                )
                chunked_reorder_est1=chunk_waveform(reordered_sources[:,0])
                chunked_reorder_est2=chunk_waveform(reordered_sources[:,1])
                chunked_mix=chunk_waveform(mix.unsqueeze(0))
                chunked_target1=chunk_waveform(sources[0].unsqueeze(0))
                chunked_target2=chunk_waveform(sources[1].unsqueeze(0))
                chunked_reorder_est_mag1= get_mag(chunked_reorder_est1).to(device)
                chunked_reorder_est_mag2= get_mag(chunked_reorder_est2).to(device)
                chunked_mix_mag= get_mag(chunked_mix).to(device)
                chunked_target1_mag= get_mag(chunked_target1).to(device)    
                chunked_target2_mag= get_mag(chunked_target2).to(device)
                
                size_= chunked_mix.shape[0]
                out_logits1=discriminator(chunked_reorder_est_mag1)
                out_logits2=discriminator(chunked_reorder_est_mag2)
                out_logits_mix=discriminator(chunked_mix_mag)
                out_logits_target1=discriminator(chunked_target1_mag)
                out_logits_target2=discriminator(chunked_target2_mag)
                out_logits1=sigmoid(out_logits1)
                out_logits2=sigmoid(out_logits2)
                out_logits_mix=sigmoid(out_logits_mix)
                out_logits_target1=sigmoid(out_logits_target1)
                out_logits_target2=sigmoid(out_logits_target2)
                out_pred_est1= (out_logits1 >= 0.5).float() # threshold 0.5 기준으로 binary 예측
                out_pred_est2= (out_logits2 >= 0.5).float() # threshold 0.5 기준으로 binary 예측
                out_pred_mix= (out_logits_mix >= 0.5).float() # threshold 0.5 기준으로 binary 예측
                out_pred_target1= (out_logits_target1 >= 0.5).float() # threshold 0.5 기준으로 binary 예측
                out_pred_target2= (out_logits_target2 >= 0.5).float() # threshold 0.5 기준으로 binary 예측
                out_pred_est1=out_pred_est1.mean().item()
                out_pred_est2=out_pred_est2.mean().item()
                out_pred_mix=out_pred_mix.mean().item()
                out_pred_target1=out_pred_target1.mean().item()
                out_pred_target2=out_pred_target2.mean().item()
                out_pred_est= (out_pred_est1 + out_pred_est2) / 2
                out_pred_mix=out_pred_mix
                out_pred_target= (out_pred_target1 + out_pred_target2) / 2
                est_disc1 = out_logits1.mean().item()
                est_disc2 = out_logits2.mean().item()
                est_disc= (est_disc1 + est_disc2) / 2
                mix_disc = out_logits_mix.mean().item()
                target_disc1 = out_logits_target1.mean().item()
                target_disc2 = out_logits_target2.mean().item()
                target_disc = (target_disc1 + target_disc2) / 2
                mix_np = mix.cpu().data.numpy() #* db2linear(-adjusted_gain, eps=0.0)
                sources_np = sources.cpu().data.numpy() #* db2linear(-adjusted_gain, eps=0.0)
                est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy() #* db2linear(-adjusted_gain, eps=0.0)
                
                utt_metrics = get_metrics(
                    mix_np,
                    sources_np,
                    est_sources_np,
                    sample_rate=args.sample_rate,
                    metrics_list=COMPUTE_METRICS,
                )
                utt_metrics["mix_path"] = test_set.mixture_path
                utt_metrics["target_logit"] = target_disc
                utt_metrics["mix_logit"] = mix_disc
                utt_metrics["est_logit"] = est_disc
                utt_metrics["out_pred_est_prob"] = out_pred_est
                utt_metrics["out_pred_mix_prob"] = out_pred_mix
                utt_metrics["out_pred_target_prob"] = out_pred_target
                
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
        default="duet",
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
        # required=True,
        # default="/path/to/dataLibri2Mix/wav16k/max/metadata",
        # default="/path/to/dataLibri2Mix/wav24k/max/metadata",
        default="duet_svs/MedleyVox",
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
