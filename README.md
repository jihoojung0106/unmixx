# UNMIXX: Untangling Highly Correlated Vocals in Multiple Singing Voice Separation

**UNMIXX** a novel framework for the multiple singing voices separation (MSVS). While similar to speech separation, MSVS presents unique challenges, namely data scarcity and highly correlated nature of singing voice. To address these issues, we propose  three key components: (1) a musically informed mixing strategy to construct highly correlated training mixtures, (2) a reverse attention that drives the two outputs apart using cross attention and (3) a magnitude penalty loss penalizing energy erroneously assigned to the other output. Experiments show that UNMIXX achieves substantial improvements, with more than ~2.2 dB SDRi gains on MedleyVox evaluation set over prior method. Audio samples are available on our [demo page](https://unmixx.github.io/).


---

## Quickstart

### 1) Data Preprocessing
We use a total of **400 hours** of singing datasets for training.  
Download the datasets and follow the preprocessing steps in [MedleyVox preprocessing steps](https://github.com/jeonchangbin49/MedleyVox/tree/main/svs/preprocess).

- [Children’s song dataset (CSD)](https://zenodo.org/record/4785016#.Y2-r2y_kFqs) — 4.9 hours  
- [NUS](https://drive.google.com/drive/folders/12pP9uUl0HTVANU3IPLnumTJiRjPtVUMx) — 1.9 hours  
- [VocalSet](https://zenodo.org/record/1193957#.Y2-tmC_kFqs) — 8.8 hours  
- [Jsut-song](https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song) — 0.4 hours  
- [Jvs_music](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music) — 2.3 hours  
- [Musdb-hq (train subset)](https://sigsep.github.io/datasets/musdb.html) — 2.0 hours  
  - Single singing regions extracted using [musdb-lyrics extension](https://zenodo.org/record/3989267#.Y2-wBOzP1qs)  
- [OpenSinger](https://github.com/Multi-Singer/Multi-Singer.github.io) — 51.9 hours  
- [K_multisinger](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=465) — 169.6 hours  
- [K_multitimbre](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=473) — 150.8 hours  



### 2) Training
Example: training with 2 GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1 \
python audio_train.py --conf_dir configs/unmixx.yml
```

---

### 3) Inference
```bash
python inference.py \
  --conf_path ckpt/conf.yml \
  --ckpt_path ckpt/best.ckpt \
  --audio_path sample_music/free_mixture.wav \
  --output_dir separated_audio
```

Outputs will be saved to `separated_audio/`.
