# Why Do Unlearnable Examples Work: A Novel Perspective of Mutual Information
This is the official repository of the ICLR 2026 paper "Why Do Unlearnable Examples Work: A Novel Perspective of Mutual Information".


## Requirements

Create environments for MI-UE:
```shell
conda create -n mi-ue python=3.9
 
pip install -r requirements.txt

conda activate mi-ue
```

## Generation
```shell
python gen_miue.py --experiment miue --poison-loss-type nt_xent_l2 --pgd-loss-type cross_entropy --post-aug --epochs 100 --batch-size 128 --perturb-iters 10 --alpha 0.2 --gpu-id 0
```


## Evaluation

### Standard Training
```shell
python evaluation_sl.py --experiment miue --backbone resnet18 --dataset cifar10 --poison-path ./gen/cifar10/resnet18/miue/poison.pt --gpu-id 0
```

### Adversarial Training
```shell
python evaluation_at.py --experiment miue --backbone resnet18 --dataset cifar10 --poison-path ./gen/cifar10/resnet18/miue/poison.pt --gpu-id 0 --epsilon 4
```

