# UNMIXX: Untangling Highly Correlated Vocals in Multiple Singing Voice Separation

**UNMIXX** a novel framework for the multiple singing voices separation (MSVS). While similar to speech separation, MSVS presents unique challenges, namely data scarcity and highly correlated nature of singing voice. To address these issues, we propose  three key components: (1) a musically informed mixing strategy to construct highly correlated training mixtures, (2) a reverse attention that drives the two outputs apart using cross attention and (3) a magnitude penalty loss penalizing energy erroneously assigned to the other output. Experiments show that UNMIXX achieves substantial improvements, with more than ~2.2 dB SDRi gains on MedleyVox evaluation set over prior method. Audio samples are available on our [demo page](https://unmixx.github.io/).


---

## Quickstart

### 1) Data Preprocessing
Follow the [MedleyVox preprocessing steps](https://github.com/jeonchangbin49/MedleyVox/tree/main/svs/preprocess).  
Make sure the preprocessed data paths match those referenced in `configs/unmixx.yml`.

---

### 2) Training
Example: training with 2 GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1 \
python audio_train.py --conf_dir configs/unmixx.yml
```

**Tips**
- Adjust `batch_size` / `num_workers` in the config to fit your GPU memory.  
- To resume training, set `resume_from_checkpoint` in your config (if supported).

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
