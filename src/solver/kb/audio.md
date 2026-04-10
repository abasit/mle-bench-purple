# Audio Knowledge Card

## Approach: spectrogram + vision model

Convert waveform to log-mel spectrogram, treat as a 2D image, train a CNN.

```python
import torchaudio
mel = torchaudio.transforms.MelSpectrogram(sample_rate=32000, n_mels=128, n_fft=2048, hop_length=512)
waveform, sr = torchaudio.load(path)
spec = mel(waveform).log()  # log-mel spectrogram → treat as image
```

Then use a timm backbone (efficientnet_b0) on the spectrogram.

## Augmentations

- **SpecAugment**: random time/frequency masking on the spectrogram.
- **Mixup** on the spectrogram domain.
- **Time shift**: roll the waveform.
- **Noise injection**: add gaussian noise at low SNR.

## Cross-validation

StratifiedKFold(5) by label. If recordings have a source/speaker ID,
use GroupKFold to prevent the same source in train and val.

## TTA

Predict on 3 random crops of the audio, average logits.

## Pitfalls

- Always resample to a consistent sample rate.
- Always `.log()` the mel spectrogram — raw mel is heavy-tailed.
- Never augment val/test data.
- Use a fixed clip length with zero-padding for shorter clips.
