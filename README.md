# Knarr: creak detection for conversational speech

Knarr scores how creaky (vocal fry) each word in a recording is, from 0 (modal) to 1 (creaky). *Knarr* is Swedish for creak. You give it
an audio file and the start and end times of the words. The package contains two detectors:

- **Knarr** (default, version 2): F4N2 fine-tuned on Swedish spontaneous dialogue, with glottal-cycle (VoiceMapping) and DeepFry frame
  targets, and trained on degraded copies of its audio (mp3, phone band, noise, reverb) so its scores hold on radio, podcast and phone
  recordings. `model="knarr_e"` gives the first Knarr (2026-11-18) for reproducing earlier results.
- **F4N2**: Joakim Gustafson's original detector (WavLM-large layers 0 + 3, three seeds, pretrained on EGG targets, fine-tuned on human
  creak labels, mostly English).

**Licence: non-commercial research use only.** The models were trained on corpora with non-commercial and research-only licences (among
them EARS, CC BY-NC 4.0, and Swedish research corpora), so the weights and anything derived from them may not be used commercially. Please
cite this repository if you use it.

## Install

You need Python 3.9 or newer with PyTorch. On a GPU machine, install the right CUDA build of PyTorch first.

```bash
pip install "knarr[gate] @ git+https://github.com/jockegustafson/Knarr.git"
# or from a clone:
git clone https://github.com/jockegustafson/Knarr.git && pip install "./Knarr[gate]"
```

`[gate]` adds `pyreaper`, which F4N2's voicing check needs; if it fails to build, install without it (Knarr does not use the check). The
first run downloads the WavLM-large speech model from Hugging Face (about 1.2 GB) once.

## Run it from Python

```python
from knarr import CreakDetector

det = CreakDetector()                    # Knarr; CreakDetector(model="f4n2") for F4N2; uses the GPU if there is one

words = [("ja", 12.31, 12.52), ("precis", 12.60, 13.14)]   # (word, start s, end s)
for r in det.score_file("dialog.wav", words, channel=0):
    print(r["label"], r["start"], r["end"], r["score"], r["status"])
```

- **Word times** are seconds from the start of the file, usually from a forced aligner (MFA, WhisperX or a Praat TextGrid). The labels are
  optional: `[(12.31, 12.52), ...]` works too.
- **Stereo dialogue:** score each speaker on their own channel (`channel=0`, `channel=1`) with their own words. `channel=None` mixes the
  channels.
- **Whole files:** give all the words of a file in one call; the file is encoded once. A 10-minute file takes a few seconds on a GPU and a
  few minutes on a CPU (`device="cpu"`; `"mps"` on Apple silicon).
- **Arrays:** `det.score(y, sr, words)` takes a NumPy array at any sample rate.
- **Frame evidence for plotting:** `t, ev = det.evidence(y, sr)` (20 ms frames; uncalibrated, not a frame probability).

## Run it from the command line

```bash
knarr dialog.wav dialog.TextGrid --tier words --channel 0 -o creak.tsv
knarr dialog.wav words.tsv --model f4n2
```

The word list is either a Praat TextGrid (its first interval tier, or the one named with `--tier`) or a TSV/CSV file with a header naming
the columns `start`, `end` and optionally `word`. The output is a TSV with one row per word: word, start, end, score, status, n_support.
To get a TextGrid back with a `creak` tier added: `python examples/score_textgrid.py dialog.wav dialog.TextGrid --tier words`.

## Try it in the browser

```bash
knarr-demo            # opens http://localhost:8765 ; --model f4n2, --device cpu|mps|cuda, --port N
```

A local page: record from the microphone or load an audio file, and get its spectrogram with the creak decisions on top (orange bands
where the score passes the threshold slider), the score curve under it, and, if you also load a TextGrid or TSV with word times, a score
per word. Click the picture to play from that point. Everything runs on your machine; nothing is uploaded anywhere. The microphone only
works when the page is opened as `http://localhost` (from another machine, load files instead). The curve scores 0.2 s windows every
50 ms with the word rule, which shows where creak is; the validated use is scoring aligned words.

## Which model, and what the scores mean

Use **Knarr** for Swedish conversation. AUC below is measured on the lab's Swedish test set: 274 words from Spontal and other Swedish dialogue corpora, labelled by ear, never used in training (1.0 is perfect, 0.5 is chance).

| Model | Trained on | Voicing check | Swedish test AUC | Women | Men |
| --- | --- | --- | --- | --- | --- |
| F4N2 | mostly English, plus EGG recordings | on | 0.80 | 0.75 | 0.84 |
| Knarr e (first version) | F4N2 + Swedish dialogue labels and glottal-cycle targets | off | 0.874 | 0.89 | 0.89 |
| **Knarr v2** (default) | Knarr e + channel augmentation | off | **0.881** | 0.90 | 0.89 |

On recordings weighted as they occur in the corpora (pool-weighted AUC), Knarr v2 reaches 0.88, Knarr e 0.85 and F4N2 0.74.

**Robustness** (198 development words, each clip degraded eight ways: mp3 at 32 and 64 kbit/s, phone band with mu-law, 4 kHz low-pass,
white noise at 20 and 10 dB, pink noise at 15 dB, reverb with RT60 0.5 s). Ranking hardly changes for either model, but Knarr e's scores
fell (on average by 0.09, by 0.19 in 10 dB noise), so with a fixed threshold it under-reported creak in noisy or phone-like audio. Knarr v2
keeps the score level (average change -0.00, mean absolute change 0.07 against 0.12).

- **Scores are word values,** not frame probabilities, and there is no calibrated threshold. 0.5 is a reasonable start; pick your own
  operating point on labelled data.
- **`INSUFFICIENT_SUPPORT`** (F4N2 only) means the word had fewer than 2 voiced frames, so F4N2 gives no score. About a third of such words
  are creaky; Knarr scores them anyway (AUC about 0.86 on them).
- **Other languages:** F4N2 was developed mainly on English; Knarr was tested only on Swedish. A multilingual Knarr (Norwegian, Danish,
  German frame targets) did not improve Swedish and was not adopted; nothing has been tested on other languages yet.

## If something goes wrong

| Problem | Fix |
| --- | --- |
| `pyreaper` will not build | install without `[gate]`; Knarr does not need it, F4N2 then needs `gate=False` |
| `ModuleNotFoundError: pkg_resources` | `pip install "setuptools<81"` (pyreaper still imports it) |
| CUDA out of memory | `CreakDetector(device="cpu")`, or split very long files |
| The WavLM download hangs | the machine needs internet access to huggingface.co once; on IPv6-less servers force IPv4, or download elsewhere and copy `~/.cache/huggingface` |

## How it was made

- F4N2: formant_tracker release `f4n2_v1` (2026-10-30); `knarr/model.py` and `knarr/features.py` are copied unchanged from its scorer,
  and `tools/check_release.py` shows the package reproduces the release's scores (r 0.99997 on 2,147 Spontal words).
- Knarr: ablation e of the KTH TTS report, section 60 (2026-11-18): F4N2's three seeds fine-tuned for 2,000 steps on Swedish human labels
  (ledger, a correction round), 23,845 agreement pseudo-labels from Spontal and SR interviews, and frame targets from VoiceMapping glottal
  cycles and DeepFry, with F4N2's EGG and non-speech losses kept. Selected on a development split; tested once.
- Knarr v2 (2026-11-19, section 60.14): as Knarr e, plus channel augmentation (half of all training windows are read from a degraded copy
  of the clip: mp3 24-96 kbit/s, phone band, 3-6 kHz low-pass, white or pink noise at 5-30 dB SNR, reverb with RT60 0.2-0.9 s, or reverb
  plus noise; the targets stay those of the clean audio), and without 28 Spontal channels (and the people on them) that are reserved as a
  test set in the F4N2 project. It replaced Knarr e by a rule fixed before training (development AUC no more than 0.01 lower, smaller score
  shift under degradation) and was then tested once. `tools/export_knarr_v.py` writes its weights.
- Weights: `knarr/weights/*.npz` with SHA-256 manifests checked at load time.

Questions and bug reports: Joakim Gustafson, KTH Speech, Music and Hearing.
