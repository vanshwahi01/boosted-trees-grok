# Diffusion memorization

Course project on when and how diffusion models start copying their training data.

`experiment/memo.py` trains a small U-Net diffusion model on a subset of MNIST. It tracks the gap between training and held-out denoising loss at each noise level, and the share of generated images that nearly copy a training image. It then compares three choices of weights at sampling time: the final checkpoint, the best single checkpoint by held-out loss, and a separate best checkpoint for each noise level.

## Run

```bash
pip install torch numpy
python experiment/memo.py --data data --n 300 --seed 0 --steps 30000 --out runs/n300_s0
```

MNIST downloads automatically into the `--data` folder. The script uses an NVIDIA or Apple GPU if one is available, otherwise the CPU.
