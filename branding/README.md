# Wayvox brand

**Wayvox** is private voice, local power.

## Positioning

Wayvox is an on-device speech-to-text and dictation stack for Linux. It combines
Whisper models, OpenVINO acceleration, Intel NPU/GPU support, and push-to-talk
desktop integrations without sending audio to a cloud service.

## Voice

- **Clear:** explain hardware and runtime behavior plainly.
- **Capable:** show that local inference can feel fast and polished.
- **Quietly technical:** keep Whisper, OpenVINO, NPU, and CUDA as implementation terms.
- **Private by default:** make local processing the headline, not a footnote.

## Visual system

| Token | Value | Use |
| --- | --- | --- |
| Ink | `#101827` | Backgrounds, dark surfaces |
| Electric cyan | `#5DE4FF` | Active state, primary accent |
| Warm coral | `#FF8066` | Recording, attention, secondary accent |
| Cloud | `#F4F7FB` | Light surfaces and text |

The mark is a `V` shaped like a voice waveform: two rising strokes frame a
central signal. It stays legible as a 16px symbolic icon and scales into a
full-color product mark.

## Compatibility note

This is a product rename with an explicit service migration. New user services
use `wayvox-*`; `make install-services` disables and removes old service units.
Environment variables, extension IDs, config paths, and model paths keep their
technical names so installed settings and model data remain usable.

## Repository assets

- `wayvox-mark.svg`: square avatar, favicon, app icon, and desktop identity mark.
- `wayvox-social-preview.svg`: 1200x630 GitHub social preview image.

Upload `wayvox-mark.svg` as the GitHub repository avatar and
`wayvox-social-preview.svg` as the social preview image. Keep both files in this
repository so future updates stay reproducible.
