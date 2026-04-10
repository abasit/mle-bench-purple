# Vision Knowledge Card

## Backbones (use timm)

| Model | timm name | Notes |
|---|---|---|
| EfficientNet-B0 | `efficientnet_b0` | Strong default, fast |
| EfficientNet-V2-S | `tf_efficientnetv2_s` | Stronger, slower |
| ConvNeXt-Tiny | `convnext_tiny` | Modern conv, good accuracy/speed |
| Swin-Tiny | `swin_tiny_patch4_window7_224` | Vision transformer |

```python
import timm
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
```

No GPU? Use `mobilenet_v3_small` with frozen features + a linear head.

## Training recipe

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
scaler = torch.cuda.amp.GradScaler()  # mixed precision — 2x speedup
```
- Batch size: largest that fits in GPU memory.
- Epochs: 10-15 (usually plateaus by epoch 8-12).
- Always `model.eval()` + `torch.no_grad()` at inference.
- Always `pin_memory=True, num_workers=4` in DataLoader.

## Augmentations (albumentations)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
train_tf = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
val_tf = A.Compose([
    A.Resize(256, 256), A.CenterCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

## Test-time augmentation (`tta:hflip`)

```python
def tta_predict(model, x):
    with torch.no_grad():
        p1 = model(x).softmax(-1)
        p2 = model(torch.flip(x, dims=[-1])).softmax(-1)  # horizontal flip
    return (p1 + p2) / 2
```

## Cross-validation

StratifiedKFold(5) by class label. If images are grouped (same patient/source),
use StratifiedGroupKFold. Average per-fold test predictions for the final submission.

## Pitfalls

- Never skip `model.eval()` at inference — BatchNorm/Dropout still active.
- Never skip `Normalize` — pretrained weights expect ImageNet stats.
- Never use ResNet18/34 by default — modern backbones are 5-10pp stronger.
- Never train without mixed precision on GPU — it's free speedup.
