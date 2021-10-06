# AAVAE

Reproducibility code for ICLR 2022 submission, AAVAE.

### Training

To train the AAVAE model

1. Create a python virtual environment.

#### Virtual environment

```
python -m venv aavae_env

source aavae_env/bin/activate

pip3 install --upgrade torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

python setup.py install
```

To reproduce the results from the paper on CIFAR-10:

```
python src/vae.py \
    --gpus 1 \
    --denoising \
    --max_epochs 3200 \
    --batch_size 256 \
    --warmup_epochs 10 \
    --val_samples 16 \
    --weight_decay 0 \
    --log_scale 0 \
    --kl_coeff 0 \
    --learning_rate 2.5e-4
```

To reproduce the results from the paper on STL-10:

```
python src/vae.py \
    --gpus 2 \
    --denoising \
    --dataset stl10 \
    --max_epochs 3200 \
    --batch_size 256 \
    --warmup_epochs 10 \
    --val_samples 16 \
    --weight_decay 0 \
    --log_scale 0 \
    --kl_coeff 0 \
    --learning_rate 5e-4
```

To reproduce the results from the paper on Imagenet:

- for imagenet, you'll also have to provide ``--data_path`` argument
- imagenet ``train`` and ``val`` folders should contain meta.bin which can be generated using the ``SSLImagenet.generate_meta_bins('/path/to/folder/with/ILSVRC2012_devkit_t12.tar.gz/')``
- ``SSLImagenet`` is present within ``src/datamodules/imagenet_dataset.py``

```
python src/vae.py \
    --gpus 4 \
    --denoising \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --max_epochs 2100 \
    --batch_size 128 \
    --warmup_epochs 10 \
    --val_samples 16 \
    --weight_decay 0 \
    --log_scale 0 \
    --kl_coeff 0 \
    --learning_rate 5e-4
```

To evaluate the pretrained encoder

- dataset can be cifar10, stl10, imagenet
- for imagenet, you'll also have to provide ``--data_path`` argument
- imagenet ``train`` and ``val`` folders should contain meta.bin which can be generated using the ``SSLImagenet.generate_meta_bins('/path/to/folder/with/ILSVRC2012_devkit_t12.tar.gz/')``
- ``SSLImagenet`` is present within ``src/datamodules/imagenet_dataset.py``

```
python src/linear_eval.py \
    --ckpt_path "path\to\saved\file.ckpt" \
    --dataset dataset \
    --data_path /path/to/imagenet \
    --gpus 4 \
    --batch_size 64
```

### Saved checkpoints

| Model | Dataset | Checkpoint | Downstream acc. |
| --- | --- | --- | --- |
| AAVAE | CIFAR-10 | [checkpoint](https://aavae.s3.us-east-2.amazonaws.com/checkpoints/aavae_cifar10.ckpt) | 87.14 |
| AAVAE | STL-10 | [checkpoint](https://aavae.s3.us-east-2.amazonaws.com/checkpoints/aavae_stl10.ckpt) | 84.72 |
| AAVAE | Imagenet | [checkpoint]() | 51.0 |
