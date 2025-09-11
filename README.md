# UNMIXX: Untangling Highly Correlated Vocals in Multiple Singing Voice Separation

**UNMIXX** is a novel framework for **multiple singing voice separation (MSVS)**.  
While MSVS is related to speech separation, it poses unique challenges:
- **Data scarcity** in multi-singer recordings  
- **Highly correlated** nature of singing voices  

To address these, UNMIXX introduces three key components:
- **Musically informed mixing** – constructs training mixtures with stronger temporal and harmonic alignment  
- **Reverse attention** – pushes the two outputs apart via cross attention, reducing residual leakage  
- **Magnitude-penalty loss** – penalizes spectrogram energy erroneously assigned to the other output  

---

### 📊 Results
- Achieves **~+2.2 dB SDRi gains** on the MedleyVox evaluation set compared with prior methods.  

---

### 🎧 Demo
🔊 [Audio samples](https://unmixx.github.io/)

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
