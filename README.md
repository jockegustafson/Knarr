# Knarr: creak detection for conversational speech

Knarr scores how creaky (vocal fry) each word in a recording is, from 0 (modal) to 1 (creaky). *Knarr* is Swedish for creak. You give it
an audio file and the start and end times of the words. The package contains two detectors:

- **Knarr** (default): F4N2 fine-tuned on Swedish spontaneous dialogue, with glottal-cycle (VoiceMapping) and DeepFry frame targets.
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

## Which model, and what the scores mean

Use **Knarr** for Swedish conversation. AUC below is measured on the lab's Swedish test set: 274 words from Spontal, and other Swedish corpora, labelled by ear, never used in training (1.0 is perfect, 0.5 is chance).

| Model | Trained on | Voicing check | Swedish test AUC | Women | Men |
| --- | --- | --- | --- | --- | --- |
| F4N2 | mostly English, plus EGG recordings | on | 0.80 | 0.75 | 0.84 |
| Knarr | F4N2 + Swedish dialogue labels and glottal-cycle targets | off | 0.87 | 0.89 | 0.89 |

- **Scores are word values,** not frame probabilities, and there is no calibrated threshold. 0.5 is a reasonable start; pick your own
  operating point on labelled data.
- **`INSUFFICIENT_SUPPORT`** (F4N2 only) means the word had fewer than 2 voiced frames, so F4N2 gives no score. About a third of such words
  are creaky; Knarr scores them anyway (AUC about 0.86 on them).
- **Other languages:** F4N2 was developed mainly on English; Knarr was tested only on Swedish. A multilingual Knarr (Norwegian, Danish,
  German) is in training.

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
- Weights: `knarr/weights/*.npz` with SHA-256 manifests checked at load time.

Questions and bug reports: Joakim Gustafson, KTH Speech, Music and Hearing.
