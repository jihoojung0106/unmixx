data preprocessing 
follow https://github.com/jeonchangbin49/MedleyVox/tree/main/svs/preprocess

training
CUDA_VISIBLE_DEVICES=0,1 python audio_train.py --conf_dir configs/unmixx.yml

inference
python inference.py \
  --conf_path ckpt/conf.yml \
  --ckpt_path ckpt/best.ckpt \
  --audio_path sample_music/free_mixture.wav \
  --output_dir separated_audio
