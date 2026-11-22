# Knarr: where creak occurs, and what the lungs have to do with it

Technical report, KTH Speech, Music and Hearing, 2026-11-22 (Knarr 1.4.0). Joakim Gustafson.

## Abstract

Creak at phrase ends is often explained by falling subglottal pressure: the lungs run low, the voice drops below its phonation threshold
and turns irregular. We test this on Swedish conversation at scale with two models that never hear the word's own voice. The first predicts
Knarr's creak decisions on 2.1 million words of 34 corpora from prosody, position and an estimated lung state; the second tracks a hidden
lung volume over 330 hours of speech from detected breaths, the air cost of speech and an air balance. Where a word sits in the phrase and
how the speech leads into it predict creak (AUC 0.78 against Knarr, held-out speakers); the lung state adds almost nothing, and across
352,000 stretches of speech the inferred lung volume does not predict creak at their ends beyond position (AUC 0.602 with and without it).
Combining the context model with Knarr does not raise Knarr's test-set AUC (0.881). The context model ships with Knarr 1.4.0 as a second
view for choosing words to label: `knarr-inspect --pick disagree` picks the words where it and Knarr disagree, as in the lab's round 4.

## 1. Background

The creak tuner of the formant_tracker project (`synth/phonation/pressure_map.py`) turns a "pressure" knob into creak: a per-frame
propensity from the talker's own prosody (F0 and energy below the talker's median, closeness to a run end, falling contours, local
irregularity) is thresholded, and the threshold falls with the knob, so creak spreads from phrase ends to the whole utterance. Its stage-2
weights were fitted on Buckeye and five other corpora with CreaPy frame labels (CreaPy ≥ 0.5 as creak; logistic regression; held-out
talkers AUC 0.70 context-only, 0.72 with irregularity), at general, style, corpus, affect and speaker level. Knarr is a better creak
detector than CreaPy on Swedish conversation (test AUC 0.88 against 0.66), and the lab now has a breath detector (Kipp) that finds the
in-breaths a lung model needs. This report retrains the propensity model with Knarr labels on Swedish speech, adds a lung state, and asks
whether either helps Knarr.

## 2. The context model (lung model v2)

**Data.** 34 Swedish corpora with Knarr v2 word scores (face-to-face and remote dialogue, radio interviews and panel shows, podcasts, a
parliament set): up to 4,000 utterances per corpus and 300 per speaker, 1–15 s, ≥ 3 words; 2,100,910 words, 3,873 persons, Knarr ≥ 0.5 on
17.6 %. Never used: the speakers held out for the Swedish creak test set (and every Spontal person with such a channel), the F4N2 project's
reserved test channels and the people on them, and any utterance holding a test-set word. Per utterance (own channel, 16 kHz, ± 0.3 s):
REAPER F0 (25–500 Hz, 10 ms), 25 ms frame energy, REAPER cycle jitter.

**Features per word**, talker-relative (the speaker's medians over all their sampled utterances):

- *own voice*: median, 10th percentile and last-50-ms F0 in semitones, energy, their slopes, voiced share, the original model's end proximity;
- *lead-up*: the same F0 / energy measures over the 0.5 s before the word, and its voiced share;
- *position*: seconds and words to the utterance end, last word, pauses before and after, duration (absolute and re the speaker), letters,
  filler, speech rate;
- *lung state* (from the breath detector's tables): air spent since the last in-breath or turn gap, the breath group's total air, air left,
  share used, time since the breath, the last breath's length and a volume estimate from it;
- *local*: REAPER jitter.

**Views.** *orig*: the original four features (logistic). *prior*: lead-up, position and lung state only, nothing of the word's own voice.
*ctx*: own-voice prosody and position; *ctx_lung*: + lung state; *full*: + jitter (boosted trees for all but orig). Five folds by person.

| View | AUC vs Knarr (held-out speakers) | Swedish test set | Round 1 | Round 3 |
| --- | --- | --- | --- | --- |
| Knarr v2 itself | – | 0.881 | 0.971 | 0.606 |
| orig (the tuner's features) | 0.796 | 0.754 | 0.590 | 0.535 |
| prior (lead-up, position, lung) | 0.779 | 0.652 | 0.658 | 0.448 |
| ctx | 0.883 | 0.788 | 0.658 | 0.507 |
| ctx_lung | 0.884 | 0.783 | 0.656 | 0.525 |
| full | 0.906 | 0.772 | 0.670 | 0.569 |

Rounds 1 and 3 were sampled where detectors disagree or are unsure, so every model is weaker there. In *ctx*, one feature carries most of
the signal: the word's lowest F0 (permutation importance 0.17 AUC), which is half a creak detector, since REAPER's F0 collapses in creak.
That is why *prior* excludes the word's own voice: it is the view that is independent of Knarr. Its strongest features are the F0 of the
lead-up (0.073) and whether it falls (0.035), the pause after the word (0.039) and the word's length (0.024); the lung variables follow at
0.0002–0.011 each (time since the breath highest).

**Levels.** As in the original, the general model can be recalibrated per corpus and per speaker (a + b × logit, ridge towards the general
model; groups ≥ 400 words and ≥ 20 creaks; 5-fold CV over utterances inside the group). Speakers differ a great deal in how much they creak:
per-speaker recalibration lowers the log loss from 0.287 to 0.241 and helps 98 % of 279 speakers; per corpus only 0.299 to 0.290.

**Combined with Knarr.** A logistic regression on Knarr's logit and the view's logit, fitted on rounds 1 + 3 and scored on the test set:
prior 0.874, ctx 0.879, ctx_lung 0.880, full 0.879, against Knarr alone 0.881. No gain. (The test set was scored for this report but nothing
was chosen on it.)

## 3. The lung state (lung model v3)

The breath tables behind section 2 were built for speech synthesis from the TTS manifest, which drops a speaker's unusable words (overlap,
laughter, a cap on host speech), and for single-channel mixes the breath detector had only checked pauses inside one manifest segment; podcasts
and radio programmes showed 6–10 detected breaths per minute of speech against 19 in Spontal. Version 3 rebuilds them from complete
own-word timelines: every word the speaker said, usable or not, and every own pause through the detector.

**Data.** 30 corpora of spontaneous Swedish conversation (dialogue, interviews, panel shows, podcasts, a quiz game; read and scripted speech
left out), per corpus up to 12 hours of a speaker's own speech, files ranked by their speech in long same-speaker stretches (≥ 20 s with no
own pause > 3 s and no more than 1 s of other speech in a pause), at most 1.5 h per speaker; ~330 hours. Breath rates rose to 11–16 per
minute in podcasts and radio. The breath detector is Kipp's (WavLM-large layers 6, 8 and 10 + a BiGRU; precision 0.87, recall 0.88);
batched with an attention mask it gives the same breaths as one clip at a time (471 of 471 on a check set).

**Air balance.** Over K consecutive breath cycles of one speaker the air breathed in must match the air spent, so the breath volume model
`log V = log s_speaker + a log t + c level + spectrum terms` is fitted on within-speaker deviations of log Σ V − log S over windows of K
breaths (S = the speech's air cost: voiced phone time + w × voiceless time, × 10^(k dB / 10)). Held-out speakers, within-speaker r:

| Breath measures | K = 1 | K = 3 | K = 5 | K = 8 |
| --- | --- | --- | --- | --- |
| length | 0.16 | 0.32 | 0.35 | 0.35 |
| length + level | 0.16 | 0.34 | 0.39 | 0.41 |
| length + level + spectrum | 0.17 | 0.38 | 0.43 | 0.45 |
| all (+ level re the pause's noise floor, disfluent) | 0.17 | 0.40 | 0.46 | 0.48 |

Best air cost w = 3, k = 0.1 per dB. V grows as t^0.54 (longer breaths are slower), 10 dB louder at the same length brings ~18 % more
air, disfluent breaths ~27 % of an ordinary one. Per corpus r is 0.42–0.58 except four podcast-like sets (0.31 and below), which
suggests editing or difficult mixes.

**Hidden lung volume.** A discrete state-space model per speaker file (forward–backward over 81 volume steps, in units of the speaker's
median breath-cycle air): speech drains, detected breaths refill by the balance model, a pause without a detected breath refills with a
learned probability that grows with its length (a silent or missed breath), a long pause relaxes the volume towards rest, and going below
the floor is penalised. Optionally, creak at the end of a stretch is evidence of a low volume (P = sigmoid(α − β V)). EM on training speakers,
5 folds. With a quarter of the detected breaths hidden, the model finds them at AUC 0.665 (pause length alone 0.615), but not their size.

**Does the lung state predict creak?** At the ends of 352,070 stretches of speech (43 % with Knarr ≥ 0.5), logistic regression, 5 folds by
speaker, the state taken without that stretch's own creak evidence:

| Features | AUC |
| --- | --- |
| stretch length, pause after, words | 0.602 |
| lung state (volume, P(low)) | 0.532 (0.562 when creak elsewhere informs the state) |
| both | 0.602 |

## 4. Rounds 3 and 4

**Round 3** (2026-11-21): 190 words sampled by Tvek's uncertainty (utterance ends of uncertain and certain utterances, fillers, medial and
random words, 15 hidden repeats) from dialogue, a three-party quiz game and podcasts. 58 creak, 127 modal, 5 unsure; repeats 15/15 the same.
Creak rates: fillers in uncertain utterances 51 %, ends of uncertain utterances 33 %, ends of certain ones 28 %, medial 28 %, random 4 %.
Tvek's p(uncertain) does not predict creak at utterance ends (AUC 0.49). Knarr reaches only 0.61 on these hard words. Training Knarr further
on the 145 usable words (ablation w) gave test 0.889 / pool-weighted 0.874 against v2's 0.881 / 0.881: within noise, not adopted.

**Round 4** (2026-11-22): 155 words from the out-of-fold predictions of section 2's *prior* view: 50 "unexpected" (Knarr ≥ 0.7, context in
its corpus's lowest 20 %), 50 "missed" (context in the top 10 %, Knarr ≤ 0.15), 30 Knarr-unsure (0.35–0.65), 15 random and 10 hidden
repeats; at most 3 per person and 1 per utterance; never within 1 s of a test-set or earlier round word. Labelled on the inspection page
(waveform with glottal pulses, spectrogram, pitch, period per cycle, half-speed playback, creaky stretches marked). Its labels go into the
next Knarr; `knarr-inspect --pick disagree` applies the same selection to any recording with the shipped context model.

## 5. The shipped context model

`knarr/weights/context_v1.json`: a one-hidden-layer net (32 ReLU units) on the *prior* features without the lung state (lead-up F0, energy,
slopes and voiced share; position; pauses; letters; filler; rate; a flag for no lead-up), trained on Knarr ≥ 0.5 for 800,000 of the words of
section 2. Held-out speakers: AUC 0.764 against Knarr; against labels 0.685 (test), 0.657 (round 1), 0.469 (round 3). `knarr/context.py`
recomputes the features from audio and word times (utterances = words split at pauses ≥ 1 s; speaker medians over the file). On 914 words its
scores correlate 0.90 with those from the training features, the difference coming from the speaker medians (one file instead of all the
speaker's sampled utterances).

## 6. Conclusions and limitations

- Where creak occurs in Swedish conversation is predicted by the prosody leading into a word and its place in the phrase, and much more by
  who is speaking. A lung state estimated from audible breaths and speech does not add to that. Either phrase-final creak is a prosodic habit
  rather than a pressure limit, or the audio-based lung estimate is too coarse to show it: the volume model explains under half of the
  variance of the air balance, and silent breaths are only partly recovered. A corpus with respiration-belt signals would settle this.
- The context model is a sampling tool, not a detector; its value for Knarr comes through labels on the words it disagrees with.
- All results are on Swedish; the corpora are broadcast and conversational, with uneven audio.

## 7. Reproduction

The analysis scripts (`lung2_data.py`, `lung2_words.py`, `lung2_fit.py`, `knarr_context_fit.py`, `lung3_tables.py`, `lung3_balance.py`,
`lung3_hmm.py`, `lung3_planner.py`, `creak_view_data.py round4`) are kept with the lab's non-public analysis code, as the corpora are not
redistributable. The shipped model and the code that uses it are in this repository.
