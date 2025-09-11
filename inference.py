#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
from pathlib import Path
from pprint import pprint

import torch
import torchaudio
import torchaudio.transforms as T
import look2hear.models

# (옵션) salience 시각화 유틸이 필요하면 주석 해제
# from basic_pitch_torch.inference import predict
# import numpy as np
# import matplotlib.pyplot as plt

# =========================
# Utils
# =========================
def load_yaml(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def derive_model_name(cfg: dict, ckpt_path: str):
    # 우선 ckpt 파일명 기준
    stem = Path(ckpt_path).stem
    # config에 audionet_name이 있으면 보조적으로 붙임
    try:
        net_name = cfg["audionet"]["audionet_name"]
        model_name = f"{net_name}-{stem}"
    except Exception:
        model_name = stem
    return model_name

def pick_model_class(cfg: dict):
    net_name = cfg["audionet"]["audionet_name"]
    if not hasattr(look2hear.models, net_name):
        raise AttributeError(f"look2hear.models에 '{net_name}' 클래스가 없습니다.")
    return getattr(look2hear.models, net_name)

def build_model(cfg: dict):
    sr = cfg["datamodule"]["data_config"]["sample_rate"]
    aud_cfg = cfg["audionet"]["audionet_config"]
    ModelClass = pick_model_class(cfg)
    model = ModelClass(sample_rate=sr, **aud_cfg)
    return model

def load_state_dict_safely(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Lightning 스타일 또는 pure state_dict 둘 다 처리
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # 바로 state_dict로 저장된 형태
        state_dict = ckpt

    try:
        # 바로 로드가 되면 끝
        model.load_state_dict(state_dict, strict=True)
        return
    except Exception:
        # prefix 정리 (예: "audio_model.")
        converted = {}
        for k, v in state_dict.items():
            if k.startswith("audio_model."):
                converted[k[len("audio_model."):]] = v
            else:
                converted[k] = v
        model.load_state_dict(converted, strict=True)

def prepare_audio_tensor(audio_path: str, target_sr: int, device: torch.device):
    waveform, original_sr = torchaudio.load(audio_path)  # [C, T]
    if original_sr != target_sr:
        resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    # [B, C, T] 로
    audio_input = waveform.unsqueeze(0).to(device)
    return audio_input, waveform, target_sr  # audio_input: [1, C, T], waveform: [C, T]

def normalize_outputs(ests):
    """
    모델 반환 타입이 tuple/dict 등 다양한 경우를 평탄화해 최종 [B, S, T] 또는 [S, T] 텐서를 반환.
    """
    if isinstance(ests, tuple):
        # 자주 쓰는 규칙 몇 가지
        if len(ests) == 2:
            ests = ests[1]
        elif len(ests) in (3, 5):
            ests = ests[0]
        else:
            ests = ests[0]

    if isinstance(ests, dict):
        # 가능한 키 이름들 통일
        for k in ["output_final", "audio_out_final", "output", "audio_out"]:
            if k in ests:
                return ests[k], ests.get("output_original", ests.get("audio_out_original", ests[k]))
        # 못 찾으면 첫 항목
        first = next(iter(ests.values()))
        return first, first

    # 텐서인 경우
    return ests, ests

def ensure_2d_channels_first(x: torch.Tensor):
    """
    [T] -> [1, T], [C, T]는 그대로
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Separate sources with Look2Hear TIGER-like model.")
    parser.add_argument("--conf_path", required=True, help="YAML config path.")
    parser.add_argument("--ckpt_path", required=True, help="Checkpoint path (.ckpt/.pth).")
    parser.add_argument("--audio_path", required=True, help="Input audio path (wav).")
    parser.add_argument("--output_dir", default="separated_audio", help="Output directory.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection.")
    parser.add_argument("--target_sr", type=int, default=None, help="Override target sample rate. (optional)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Device] {device}")

    # Config
    config = load_yaml(args.conf_path)
    print("[Config] loaded.")
    # pprint(config)

    # Model
    model = build_model(config)
    load_state_dict_safely(model, args.ckpt_path)
    model.to(device).eval()

    # Target SR
    cfg_sr = config["datamodule"]["data_config"]["sample_rate"]
    target_sr = args.target_sr if args.target_sr is not None else int(cfg_sr)
    print(f"[SampleRate] target_sr={target_sr}")

    # Audio
    print(f"[Audio] loading: {args.audio_path}")
    audio_input, waveform_cpu, sr = prepare_audio_tensor(args.audio_path, target_sr, device)
    print(f"[Audio] shape (B,C,T)={tuple(audio_input.shape)}, sr={sr}")

    # Forward
    with torch.no_grad():
        outs = model(audio_input, istest=True)

    ests_speech, ests_speech_original = normalize_outputs(outs)
    # [B, S, T] 또는 [S, T]
    if ests_speech.dim() == 2:
        # [S, T] -> [1, S, T]
        ests_speech = ests_speech.unsqueeze(0)
    if ests_speech_original.dim() == 2:
        ests_speech_original = ests_speech_original.unsqueeze(0)

    # 첫 배치만 사용
    ests_speech = ests_speech[0].cpu()             # [S, T]
    ests_speech_original = ests_speech_original[0].cpu()  # [S, T]
    num_speakers = ests_speech.shape[0]
    print(f"[Separation] detected {num_speakers} streams")

    # Output paths
    model_name = derive_model_name(config, args.ckpt_path)
    base_dir = Path(args.output_dir) / model_name / Path(args.audio_path).stem
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save estimates
    for i in range(num_speakers):
        # torchaudio.save expects [C, T]
        est = ensure_2d_channels_first(ests_speech[i])
        est_org = ensure_2d_channels_first(ests_speech_original[i])
        out_path = base_dir / f"spk{i+1}.wav"
        print(f"[Save] {out_path}")
        torchaudio.save(str(out_path), est, sr)

    # Save mixture, too
    mix_out = base_dir / "mixture.wav"
    print(f"[Save] {mix_out}")
    torchaudio.save(str(mix_out), waveform_cpu, sr)

    print("[Done] All files saved under:", base_dir)

if __name__ == "__main__":
    main()
