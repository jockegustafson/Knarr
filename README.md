# Knarr

Word-level **creak (vocal fry) detection for conversational speech**. Knarr scores how creaky each word sounds, from 0 (modal) to 1
(creaky), from the audio and the words' start and end times. On a held-out Swedish dialogue test set (274 words labelled by ear) it
reaches **AUC 0.88**, against 0.80 for F4N2, 0.79 for DeepFry and 0.66 for CreaPy on the same words, and its scores stay level on mp3,
phone-band, noisy and reverberant audio. A 10-minute file takes a few seconds on a GPU. *Knarr* is Swedish for creak.

```bash
pip install "knarr[gate] @ git+https://github.com/jockegustafson/Knarr.git"
knarr dialog.wav words.tsv -o creak.tsv      # word list as TSV (word, start, end) or a Praat TextGrid (--tier words)
knarr-demo                                   # local web page: record or load audio, see the creak on a spectrogram
knarr-inspect dialog.wav words.tsv -o check.html  # a page for checking words by ear and eye (waveform, pulses, half speed)
```

```
word     start  end    score  status
Hej,     2.580  3.600  0.00   ok
förstår  4.460  4.900  0.00   ok
säger.   5.060  5.460  0.59   ok
här      5.900  6.040  0.00   ok
ut.      6.340  6.680  0.71   ok
```

Five of the 17 words of a 7-second Swedish recording: the two phrase ends, *säger.* and *ut.*, are creaky, the rest is modal.

The package contains two detectors:

- **Knarr** (default, version 2): F4N2 trained further. F4N2's own labelled words and EGG data stay in its training, and Swedish
  spontaneous dialogue (Spontal, SR interviews) is added, with glottal-cycle (VoiceMapping) and DeepFry frame targets. It is also trained
  on degraded copies of its audio (mp3, phone band, noise, reverb), so its scores hold on radio, podcast and phone recordings.
  `model="knarr_e"` gives the first Knarr (2026-11-18) for reproducing earlier results.
- **F4N2**: Joakim Gustafson's original detector (WavLM-large layers 0 + 3, three seeds). It was pretrained on EGG recordings in English,
  German and Swedish and fine-tuned on words labelled by ear, about 90 % of them English, plus Swedish words and Danish stød.

The corpora behind each model are listed under [Training data](#training-data).

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
knarr dialog.wav words.tsv --context -o creak.tsv       # adds a `context` column (needs pyreaper)
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

## Check words by ear and eye

`knarr-inspect` writes one self-contained web page for labelling creak by hand: open it in any browser, nothing is uploaded.

```bash
knarr dialog.wav words.tsv -o creak.tsv                          # score the words first (optional)
knarr-inspect dialog.wav creak.tsv --scores creak.tsv --pick uncertain --show-score -o check.html
```

Each word is shown on one time axis (0 = word start, the word ± 0.25 s) in four views:

- **waveform**, zoomable down to single glottal cycles, with REAPER's glottal pulse marks;
- **spectrogram**, wideband (pulses as vertical striations; creak spaces them widely and unevenly) or narrowband (harmonics);
- **pitch**: REAPER's F0 and the pulse rate (1 / interval) on a log scale;
- **period of every cycle**, with cycles that differ by more than 20 % from the previous one marked.

You can play the word, the word at **half speed** (an octave lower, so single pulses become audible), the 2.5 s before it, a loop, or any
stretch you drag across. Labels are creaky (any part of the word) / not creaky / unsure / no speech (noise or a wrongly timed word, left out), an optional creak type (phrase creak, glottal onset or stop,
breathy-creaky) and any number of marked creaky stretches inside the word. They are kept in the browser; **Download labels (JSON)** saves
them as `[{code, word, src: {audio, start, end}, label, type, spans: [{s, e}]}]`, with span times in seconds from the word start.

- `--pick uncertain` keeps the words Knarr scored 0.3–0.7, the ones worth a human ear; `--pick creaky` keeps those ≥ 0.5; `--pick disagree`
  keeps the words where Knarr and the context model disagree (see below). All need `--scores`.
- `--show-score` shows Knarr's score only after you have labelled the word, so it does not steer you.
- `--max` (default 200) caps the number of words, drawn at random (`--seed`); `--channel` and `--tier` work as for `knarr`.
- `--parts N` writes the words to N script files beside the page (`<page>_items_<k>.js`), for web hosts that cap the size of one file
  (300 words with their clips come to about 15 MB); keep the files together.
- With Huanchen Cai's [VoiceMapping](https://github.com/HuanchenCai/VoiceMapping) installed (importable as `voicemap`), its audio-only
  IAIF glottal cycles are added as a second pulse layer. It needs pyreaper (`pip install "knarr[gate]"`) and uses ffmpeg for the
  listening clips when it is on the PATH.

## Which model, and what the scores mean

Use **Knarr** for Swedish conversation. AUC below is measured on the lab's Swedish test set: 274 words from Spontal and other Swedish dialogue corpora, labelled by ear, never used in training (1.0 is perfect, 0.5 is chance).

| Model | Trained on | Voicing check | Swedish test AUC | Women | Men |
| --- | --- | --- | --- | --- | --- |
| F4N2 | labelled words, mostly English; EGG in 3 languages | on | 0.80 | 0.75 | 0.84 |
| Knarr e (first version) | F4N2's data + Swedish dialogue (labels, pseudo-labels, glottal-cycle targets) | off | 0.874 | 0.89 | 0.89 |
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
- **Other languages:** F4N2's labels are mostly English, and its direct predecessor F4 was tested on unseen English speakers (EARS, AUROC 0.87);
  Knarr was tested only on Swedish. A multilingual Knarr (Norwegian, Danish,
  German frame targets) did not improve Swedish and was not adopted; nothing has been tested on other languages yet.

## Where creak is expected: the context model

`knarr.context` scores how likely creak is on a word from **what leads into it alone**, never from the word's own voice: the 0.5 s
before it (F0 and energy relative to the speaker, and whether they fall), the pauses before and after it, its position in the utterance,
its length, whether it is a filler, and the speech rate. It was trained to predict Knarr's decisions on 2.1 million words of 34 Swedish
corpora and reaches AUC 0.76 against Knarr on held-out speakers (0.69 against the Swedish test set's labels, where Knarr reaches 0.88).
It is not a creak detector. It is a second, independent view: the words where it and Knarr disagree are the most informative ones to
check by ear (`knarr-inspect --pick disagree`): creak Knarr hears where nothing in the context leads one to expect it (noise, laughter,
or real but unusual creak), and expected creak that Knarr does not hear.

```python
from knarr.context import context_scores
context_scores("dialog.wav", [("ja", 12.31, 12.52), ("precis", 12.60, 13.14)], channel=0)   # one value per word, 0-1
```

Adding the context score to Knarr's own score does not raise the test-set AUC (0.874-0.880 against 0.881), and neither did a model of the
speaker's lung state built from detected breaths: across 330 hours of Swedish conversation, how much air is left does not predict creak
beyond where the word sits in the phrase. The study is in [docs/REPORT.md](docs/REPORT.md).

## Training data

Knarr is not trained on Swedish dialogue alone: every Knarr version keeps F4N2's labelled words, EGG targets and non-speech negatives in
its training and adds the Swedish material. Labels are always the owner's judgements by ear; no detector output is used as a label, except
the Swedish pseudo-labels, which are words where F4N2 and the glottal-cycle view agree.

| Data | Corpora (language) | Size | F4N2 | Knarr |
| --- | --- | --- | --- | --- |
| Words labelled by ear | EARS 488, DAPS 34, AVID 30, TAVA 29, Buckeye 27, CMU ARCTIC 27, MSP-Podcast 17 (English); TINA 36, Martin 10 (Swedish); Joe 12 | 710 judged words, 653 with a clear label | yes | yes |
| Stød labels | DanPASS audit (Danish) | 71 units | yes | yes |
| EGG targets (pretraining; a side loss in fine-tuning) | AVID, CMU ARCTIC, PTDB-TUG, TAVA (English), Martin (Swedish), Saarbrücken healthy (German) | 3,000 pretraining steps | yes | AVID, CMU ARCTIC, TAVA, Saarbrücken |
| Non-speech negatives | DanPASS pauses, ledger silences | – | yes | yes |
| New Swedish labels | DanPASS 74 and Spontal 50 ledger items; a correction round, Spontal 103 and SR interviews 105 | 332 words | – | yes |
| Pseudo-labels | Spontal, SR interviews (Swedish) | 23,845 words | – | yes |
| Glottal-cycle and DeepFry frame targets | Spontal, SR interviews (Swedish) | 10,549 clips + 12,808 utterances (26.6 h) | – | yes |
| Degraded copies (v2) | the Knarr clips above, mp3 / phone / low-pass / noise / reverb | one copy per clip | – | v2 only |

Tried and not adopted: Buckeye's own creak tags, and Norwegian (NB Samtale), Danish (CoRal) and German (Emilia) frame targets. The
corpus links and licences are in the F4N2 report (`formant_tracker/docs/F4N2_CREAK_DETECTOR_REPORT.md`, section 3.2).

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
- Knarr: ablation e of the KTH TTS report, section 60 (2026-11-18): F4N2's three seeds trained for 2,000 more steps on F4N2's own 653
  labelled words and 71 DanPASS stød units plus the Swedish labels (ledger, a correction round), 23,845 agreement pseudo-labels from
  Spontal and SR interviews, and frame targets from VoiceMapping glottal cycles and DeepFry, with F4N2's EGG and non-speech losses kept. Selected on a development split; tested once.
- Knarr v2 (2026-11-19, section 60.14): as Knarr e, plus channel augmentation (half of all training windows are read from a degraded copy
  of the clip: mp3 24-96 kbit/s, phone band, 3-6 kHz low-pass, white or pink noise at 5-30 dB SNR, reverb with RT60 0.2-0.9 s, or reverb
  plus noise; the targets stay those of the clean audio), and without 28 Spontal channels (and the people on them) that are reserved as a
  test set in the F4N2 project. It replaced Knarr e by a rule fixed before training (development AUC no more than 0.01 lower, smaller score
  shift under degradation) and was then tested once. `tools/export_knarr_v.py` writes its weights.
- Round 3 (2026-11-21): 190 words chosen by uncertainty (the certainty detector Tvek; dialogue, podcast and quiz-game speech), 15/15
  hidden repeats labelled the same. A Knarr trained on v2's data plus these 145 new words (ablation w) reached 0.889 on the test set but
  0.874 pool-weighted (v2: 0.881 / 0.881): within noise, not adopted.
- Context model (1.4.0, 2026-11-22): `knarr/weights/context_v1.json`, a one-hidden-layer net on the lead-up features, trained on Knarr v2's
  decisions (>= 0.5) for 800,000 words sampled from 2.1 M words of 34 Swedish corpora (speakers held out for the test set and the F4N2 test
  channels never used); details and the lung-state study in [docs/REPORT.md](docs/REPORT.md).
- English (1.4.1, 2026-11-26): Knarr v2 and F4N2 measured on 63,178 aligned words of acted emotional (ESD), podcast (MSP-Podcast) and
  read (MAGES) English; a 291-word labelling round from the words where Knarr is unsure is under way ([docs/REPORT.md](docs/REPORT.md),
  section 7). The inspection page no longer grows its views on high-density screens.
- Weights: `knarr/weights/*.npz` with SHA-256 manifests checked at load time.

Questions and bug reports: Joakim Gustafson, KTH Speech, Music and Hearing.
