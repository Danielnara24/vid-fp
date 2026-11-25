# vid-fp on the generated corpus: first measurement (2026-11-18)

Binary: target/release/vid-fp 0.40.0 (static, commit 522b7ab). Corpus: /home/daniel/Documents/VIDS.
Score files: benchmark/corpus/out/d*-p*.score.txt, FP/MISS lists in out/d*-p*.dump.

## Corpus

- 51 seeds (3.4 h), 1,150 derived files, 1,201 files in total, 61 h of video, 46.6 GB.
- Ground-truth pairs: 10,810 SAME, 5,359 CLIP, 900 PARTIAL, 325 DIFFERENT. Every other pair is a negative.
- Each seed got a deterministic subset of the 100 transforms, because all of them on every seed would take about 26 h. The rates were:
  - container: always
  - mirror: 35%
  - clip: 30%
  - clip_chain: 25%
  - everything else: 12%

  Seeds over 15 minutes (mit_linalg) only get clips and remuxes. `--all` gives the full set. talkhead_1 was the test seed and has all 99 non-failing transforms, so it is over-represented. See the last split below.
- Generation took 4.5 h at 6 workers. Two generator bugs were found and fixed:
  - `reaction` had an endless generated source and hung;
  - MOV cannot hold AV1/VP9, so `remux_mov` writes MP4 for those seeds.
- Seed problems:
  - Two URLs are gone: Tom & Jerry, so the shared-intro set is lost, and football 2, so football 1 was split in two.
  - Spring was listed twice in urls.txt and is one seed.

## Headline, `-d 4 -p 20` (default)

Cold scan: 135 s wall, 935 s user, 194 MB peak RSS, exit 0.

| | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| all pairs | 8327 | 2209 | 7166 | 0.790 | 0.538 | 0.640 |
| without talkhead + podcast seeds | 5077 | 62 | 5346 | 0.988 | 0.487 | 0.653 |
| without mirror files | 8188 | 2179 | 3433 | 0.790 | 0.705 | 0.745 |
| without either | 4969 | 61 | 2599 | 0.988 | 0.657 | 0.789 |
| without talkhead_1's family | 5508 | 447 | 5633 | 0.925 | 0.494 | 0.644 |

Seed to derived recall (would it find my original?): 753 / 1123 = 0.671.

### False positives: 2,209

| kind | links |
|---|---|
| near-miss talkhead | 1,849 |
| near-miss podcast | 262 |
| near-miss news | 36 |
| same seed, disjoint clips | 34 |
| unrelated seeds | 16 |
| near-miss starship | 12 |

### Merges

11 seed pairs are joined by false links, and 13 of the 109 groups hold one.

- **talkhead_1/2/3 and podcast_1/2 match each other as seeds.** These are pieces of one static-camera video: same person, set and framing. vid-fp matches 82 of talkhead_2's 150 s against talkhead_1, and 179 of 200 s between podcast_1 and podcast_2.
  - Through the keyframe DCT hash these are near enough the same picture, so the FPs are the hash's resolution limit, not noise.
  - This is the near-miss design working as intended, and it is the main precision problem. At `-p 80` precision is 0.935.
  - Whether a static same-set split *should* be a negative is worth a decision. The alternative is to treat these sets as their own tier, scored separately like traps.
- **news_1 + news_2, 36 links.** They are from the same channel and probably share a graphic or bumper. Worth checking the frames.
- **starship_1 + starship_3, 12 links.** The news upload probably rebroadcasts the drone feed, which would make these genuinely shared footage. It needs checking before they are counted as errors.
- **Unrelated seeds, 16 links:**
  - lyric_black with talkhead_1, spring and underwater: mostly black frames;
  - podcast_1 with talkhead_1, 7 links;
  - beach_2 with wow_2, 1 link.

## Recall by group, seed to derived

| group | found |
|---|---|
| container | 255/255 |
| encoding | 103/110 |
| colour | 92/106 |
| resolution | 42/60 |
| clip | 144/207 |
| overlay | 42/64 |
| clip_chain | 46/92 |
| pipeline | 13/31 |
| geometry | 10/19 |
| crop | 6/15 |
| **mirror** | **0/164** |

Transforms at 0:
- every mirror variant;
- vertical_reframe;
- clip_vertical;
- blur_fill;
- crop_zoom;
- vhs;
- tiktok;
- reaction;
- pip;
- screen_filmed.

Weak transforms:
- clip_crop 1/13;
- compilation 8/30;
- clip_tiny 7/19 (2.6 to 4.5 s clips);
- screen_recording 2/7;
- lower_third 2/5;
- tv_capture 3/7;
- clip_mid 10/18.

Missed clips include long ones, such as a 120 s clip_mid of blender_tut_2 and a 162 s clip_mid of calculus_3b1b. A length problem does not explain those. The likely cause is keyframe sampling: the clip's re-encoded GOP lands on different frames than the YouTube seed's keyframes. That is worth checking first.

## Ladder at `-p 20` (warm cache)

| -d | precision | recall | F1 | seed recall | seed pairs merged |
|---|---|---|---|---|---|
| 2 | 0.808 | 0.485 | 0.606 | 0.620 | 9 |
| 4 | 0.790 | 0.538 | 0.640 | 0.671 | 11 |
| 6 | 0.775 | 0.570 | 0.657 | 0.704 | 13 |
| 8 | 0.766 | 0.591 | 0.667 | 0.723 | 16 |
| 10 | 0.751 | 0.606 | 0.671 | 0.733 | 37 |
| 12 | 0.730 | 0.626 | 0.674 | 0.750 | 107 |

At `-d 4`, `-p 50` gives P 0.799 and R 0.452, and `-p 80` gives P 0.935 and R 0.330.

Warm comparisons take 1 to 4 s. The cold scan was 135 s for 61 h of video. No performance problem showed up.

## Where to start

1. **Mirror (0/164).** The hash is not flip-invariant. The cheapest route is a second, flipped hash per keyframe probed in phase 1. The DCT of a horizontally flipped image flips the sign of the odd horizontal frequencies, so the flipped hash can be derived from the coefficients without a second decode.
2. **Clips** whose GOP does not line up with the seed's keyframes, and the 3 s clips.
3. **Reframes and crops:** vertical_reframe, clip_vertical, crop_zoom, blur_fill, pip, reaction. These are large spatial changes the global hash cannot see.
4. **Precision** is almost entirely the static same-set near-misses. Decide how to score them before tuning against them.

## Follow-up: false-positive frames and why clips are missed (2026-11-18)

Matched frames for one link of each of the 18 seed pairs joined at `-d 4 -p 20`:
- contact sheets: out/evidence/NN.jpg;
- page: https://claude.ai/artifact/GZts699tSxgq7xAqf2kVpS;
- extracted by the uncommitted examples/evidence.rs, which wraps the window's `compare::evidence`.

Each verdict below needs a decision from you:
- **Shared footage, so arguably not false.**
  - news_1 + news_2: the Ingraham and Vance segments are re-aired, with a 0-bit match.
  - starship_1 + starship_3: starship_3 opens with about 6 s of the same drone feed, confirmed by alignment.
- **Wrong matches (a real precision bug).** The three lyric_black links: text or a logo on black, against other text or a logo on black.
- **Same scene at a different moment.** This covers:
  - talkhead_1/2/3 and podcast_1/2;
  - a compilation holding podcast_2;
  - beach_2 with wow_2's compilation;
  - every same-seed disjoint pair.

  These are 2,190 of the 2,209 false links.

### Why clips are missed: two causes

1. **Keyframe timing.** Across 269 clip and clip_chain pairs, the median distance from a clip keyframe to the nearest seed keyframe is 1–2 s. Recall by keyframe position:

   | | found |
   |---|---|
   | median offset < 0.1 s | 28/30 (0.93) |
   | median offset ≥ 0.1 s | 154/239 (0.64) |
   | same crop, keyframes aligned | 25/26 (0.96) |
   | same crop, keyframes off | 140/196 (0.71) |

   Off-grid clips survive only when the picture barely changes in 1–2 s (beach, news, talking heads). High-motion footage fails: F1, underwater, ds3, anime, dashcam, football, concert.

   Control: the f1_onboard clip_mid re-encoded with keyframes forced onto the seed's matches 100%, against 0% with a normal GOP. underwater goes from 19% to 84%.

   The hash is not the weak point. One frame stays within 0–4 bits at 360p, at 240p with CRF 28, and through a re-encode. But a frame one sample (27 ms) off on screen content is already 8–10 bits apart.
2. **The crop rectangle is chosen per file** (`hash_frames`). The outer band is cropped wherever per-pixel variance across all of that file's keyframes is ≤ 2. A clip sees fewer frames than its seed and often gets a tighter crop, which reframes every hash:
   - ds3: seed rows 0–63, clip 8–61;
   - calculus: seed 0–63 × 0–59, clip 5–60 × 2–58.

   Pixel-identical frames end up 22 bits apart. Pairs whose crop differs from the seed's are found 17/47 (0.36), against 165/222 (0.74) when it matches.

## Step 0 and the keyframe-timing investigation (2026-11-18, second session)

### Scoring changes (score.py)
- **SCENE tier:** same seed, disjoint stretches; talkhead_1/2/3 and podcast_1/2 (SCENE_SETS); and compilations built from them. Excluded from P/R and reported as "scene links k of n".
- **Shared footage:** links between the news_1/news_2 and starship_1/starship_3 families are counted on their own line and kept out of P/R. The shared intervals could not be labelled automatically: align_shared.py found the news run but not the colour-graded starship feed.
- **Baseline re-scored at `-d 4 -p 20`:**
  - precision 0.999, recall 0.538;
  - 2,153 of 4,486 scene pairs linked; 48 shared-footage links.
  - The 8 remaining FPs are all lyric_black.

| -d (`-p 20`) | P | R |
|---|---|---|
| 2 | 0.9997 | 0.485 |
| 6 | 0.998 | 0.570 |
| 8 | 0.995 | 0.591 |
| 10 | 0.981 | 0.606 |
| 12 | 0.947 | 0.626 |

### What a frame between keyframes costs
- Measured on 24 random corpus files (out/cost.txt) by decoding every frame with ffmpeg on one thread, against vid-fp's own keyframe-only fingerprint.
- **Full decode costs 25.6x keyframes at the median, from 5x to 178x, and 28.4x summed.**
- `skip_frame noref` helps H.264/HEVC re-encodes about 2x and AV1/VP9 not at all.
- `skip_loop_filter` saves 9% on H.264 and nothing on AV1 (libdav1d ignores it).
- On 1080p60 AV1 seeds a keyframe costs about the same as an inter frame.
- Any frame between keyframes costs a decode of everything from the previous keyframe up to it, so a moment chosen by content costs a dense decode.

### How exact "exact" has to be
- vid-fp's hash moves 2–4 bits for one frame of offset, 4–6 for two, and 8–12 for five (anime, underwater: out/offset_bits.py).
- So samples have to land within about one frame.

### Simulation: 94 clip pairs of 16 seeds
- Tool: sim.py, dense 64x64 tracks, vid-fp's hash with no crop, strict `-d 4`, clip side >= 20%.
- Each rule sees only one file.

| rule | found | same moment |
|---|---|---|
| keyframes (today) | 45 | 0.07 |
| keyframe + 6 frames, both files | 59 | 0.13 |
| keyframe + 12 | 62 | 0.17 |
| keyframe + 18 | 69 | 0.21 |
| keyframe + 12 frames spread over 24 | 69 | 0.24 |
| oracle: clip keyframes against every seed frame | **83** | 0.97 |

- Peaks of frame-to-frame change (a content-defined rule) found 4/5 against 1/5 on anime, but it needs every frame decoded.
- **Cuts are not enough:** the failing clips are mostly continuous footage (calculus 0 cuts in 162 s, f1 and football 1 each), and encoders keyframe only about a quarter of the cuts there are.

### Prototype: keyframe runs in vid-fp
- Branch exp/keyframe-runs, uncommitted.
- `VIDFP_RUN=N` decodes N packets after every keyframe. Each run frame is timed on the keyframe's clock and stands for the keyframe's span, and a span counts once if any of its hashes matched.

| run length | wall (cold) | user CPU | P | R | seed recall | clip | clip_chain |
|---|---|---|---|---|---|---|---|
| 0 (baseline) | 135 s | 935 s | 0.999 | 0.538 | 0.671 | 144/207 | 46/92 |
| 12 | **1181 s (8.7x)** | 8611 s | **0.908** | 0.602 | 0.744 | 177/207 | 64/92 |

- **The precision loss is almost entirely lyric_black:** 933 links to unrelated seeds, as text-on-black run frames match everything.
- Seven .ogv (Theora) files failed: a prototype bug.

### Conclusion so far
Exact moments need a dense decode of at least one side of a pair:
- **Every file dense:** about 28x.
- **Only the cheaper side of candidate pairs** (pairs with weak keyframe evidence): 2.5–5x extra. But that tops out at recall 0.58–0.65, because most missed pairs have no keyframe evidence to be picked up by.
- **What fits in 4x:** runs of 2–6 frames. They raise recall by finding *more* shared moments, not the *same* ones.

### Two no-decode tests (2026-11-18)

**Test 2: compressed frame sizes recover a clip's offset (out/sizes.py). No.**
- Bytes per bin read from packet headers, log-scaled, with keyframes kept, capped or dropped, detrended, then cross-correlated. Bins of 0.1, 0.25 and 0.5 s were tried.
- Best setting: within 1 s for 25/67 clips, within one bin for 22/67.
- A single clear peak is right only about half the time.
- Sizes follow each encoder's rate control, not the picture.

**Test 1: a coarser keyframe descriptor that tolerates the 1–2 s gap (out/coarse.py, out/coarse_all.py). No.**
- Against 16 seeds it looked strong: a 4x4 grid found 82/94 clips against 71 for the hash.
- Against the whole corpus it collapses. Each of the 94 clips was scored against the keyframes of all 1,201 files (111,445 judged pairs, 3,119 positive), with intra-only files thinned to 2 keyframes/s.
- Recall at a fixed precision, with each threshold swept for its best:

| descriptor | P >= 0.995 | P >= 0.99 | P >= 0.98 |
|---|---|---|---|
| vid-fp hash, 20% of keyframes within tau bits | 0.586 (tau 10) | 0.635 (12) | 0.654 (14) |
| grid 8x8, normalised | 0.580 | 0.589 | 0.596 |
| grid 4x4, normalised | 0.396 | 0.425 | 0.498 |
| histogram 16 | 0.319 | 0.319 | 0.429 |
| vid-fp itself at `-d 4 -p 20` (with corroboration) | 0.587 at P 0.999 | | |

- Every coarse descriptor that survives the gap also matches unrelated footage.
- The 64-bit hash with corroboration is already the best keyframe-only option, and loosening it trades precision for recall at about the rate the `-d` ladder shows.

## Per-frame crop and mirror matching (2026-11-18, branch fix/crop-mirror, uncommitted)

- **Per-frame crop (`frame_crop`):** each frame peels its own flat edge bars.
- **Mirror matching:** a mirrored hash is stored per sample, computed from the same DCT coefficients, and every pair is measured both ways round.
- **Cache:** the table is renamed to `fingerprints_dct_ct_ff8_framecrop`.
- **Checks:** tests pass (402 library tests plus integration), and clippy is clean.

| `-p 20` | P before | R before | P after | R after |
|---|---|---|---|---|
| `-d 2` | 0.9997 | 0.485 | 0.9985 | 0.595 |
| `-d 4` | 0.9990 | 0.538 | 0.9963 | 0.670 |
| `-d 8` | 0.995 | 0.591 | 0.987 | 0.753 |
| `-d 12` | 0.947 | 0.626 | 0.929 | 0.805 |

### At `-d 4`
- **Seed recall:** 0.671 -> 0.767.
- **Mirror:** 2/164 -> 102/164. Plain mirror is now 17/18. mirror_crop, mirror_tilt and mirror_vertical stay low: those are spatial reframes.
- **Crop alone:** recall 0.538 -> 0.541. Clip pairs whose per-file crop differed from the seed's went 17/47 -> 26/47. The rest are spatial transforms (clip_vertical, clip_crop) or keyframe timing.
- **False positives:** 39, all lyric_black and starship_2 text cards.

### Cost
Master, timed under the same machine conditions, as the earlier 135 s baseline no longer reproduces:

| | master | this branch |
|---|---|---|
| cold scan | 252 s | 262–272 s (+4–8%) |
| warm comparison | 0.75 s | 1.27 s |

## Matching after decode: detail hash, reframings, adaptive window (2026-11-18, branch exp/match-lab, uncommitted)

Lab: `fps.py`, `views2.py`, `hash2.py`, `rules7.py`/`rules8.py` replay vid-fp on a dump of every sampled 64x64 keyframe (out/frames, ~1.2 GB); the lab matched the binary to within a few pairs. Held-out check: seeds split in two by crc32, each half scored alone (`rules3.halves`).

| change | why it generalises | effect |
|---|---|---|
| featureless floor 8 -> 50 | real keyframes' 1st percentile ~60 | FP 46 -> 7 at `-d 4`, -19 TP |
| detail hash (mid-band of a 32x32 DCT), distance = main + detail over 128 bits, same sigma rule | no new constant: -d keeps its sigma meaning | text-on-plain and layout look-alikes stop matching |
| band reframing (portrait frame's middle 16:9) | the standard vertical re-upload layout | vertical fills 0/50 -> most, no FP cost |
| zoom 1/1.12 reframing | light zoom/crop/tilt-to-fill | +~0.03 R |
| alignment window = larger median keyframe gap of the pair | offset error is one keyframe gap | beats 500/1000/2000 ms fixed |
| default `-d` 4 -> 6 | rung where nothing recalls worse than before | |

Binary, `-p 20`, cold scan 294-337 s (master 252 s measured earlier), warm ~10 s:

| | P | R | seed recall |
|---|---|---|---|
| fix/crop-mirror `-d 4` | 0.9963 | 0.670 | 0.767 |
| new `-d 4` | 0.9995 | 0.778 | 0.850 |
| new `-d 6` (default) | 0.9991 | 0.802 | 0.866 |
| lab `-d 7` | 0.9965 | 0.823 | |
| lab `-d 8` | 0.993 | 0.837 | |

Measured and refused: vertical strips (3 positions; +71 TP, +144 FP), union of narrow and wide rules (same frontier, lower P), pair-level evidence gates, witness self-separation, z125 (more FP than TP at the same P). Gap fill along an offset-consistent chain: +0.017 R for +5 FP at `-d 6`, but needs a fitted gap constant, not shipped. Not reachable by matching: pip/reaction/screen_recording/screen_filmed (picture inset in a composite), vertical crops, clips whose keyframes share no moment with the source.

### Rescue and inset (2026-11-19, committed on exp/match-lab)
- Aligned rescue (one more slack step inside a chain of >= 2 anchors): `-d 6` R 0.801 -> 0.823, P 0.9991 -> 0.9986 (added FPs all lyric_1/2 shared intro).
- Inset reframing (largest edge-bounded rectangle): R 0.823 -> 0.839, P 0.9986, cold 310 s, warm 15 s. Robust across step 8/12/16 and cover 0.7/0.8/0.9 (R 0.839-0.859, FPx 9-15).
- Refused: detail crop (peel low-detail lines; +0.009 R), vertical strips under the wide rule (centre strip +39 TP/+1 FP; three strips +59/+91).
- Chain-envelope fill (mark every sample between a chain's first and last anchor): chain>=2 R 0.846 -> 0.867 at P 0.9956 (below the line); chain>=3 R 0.857 at P 0.9973; chain>=4 R 0.850. Not shipped: the anchor count would be chosen by this corpus's precision line.
- Compilation misses (271 pairs): 120 measured under the 20% gate, 150 never matched; blur-fill misses are hard content (lectures, UI, lyrics, b/w film, dark ocean) at ~20 rows of band.

## 128 working frame, strips, 4:3 cut, captioned bars, span credit; default -d 7 (2026-11-19)

Scored two ways: the original scorer, and `SHARED_LYRIC=1`, which treats lyric_1/lyric_2 as shared footage (the user confirmed they share a real intro), like news_1/2 and starship_1/3. Baseline under it: committed 64 build `-d 6` P 0.9993 R 0.839 (original scorer P 0.9986).

Each step, `-d 7 -p 20`, SHARED_LYRIC scorer unless noted:

| step | P | R |
|---|---|---|
| 64 build, `-d 6` (committed) | 0.9993 | 0.839 |
| 128 working frame, `-d 6` | 0.9998 | 0.864 |
| 128, `-d 7` | 0.9997 | 0.881 |
| + strips (7 positions) | 0.9994 | 0.885 |
| + tiny frames featureless + chained span credit | 0.9994 | 0.889 |
| + 13 strip positions | 0.9993 | 0.894 |
| + 4:3 centre cut (shape-restricted) | 0.9993 | 0.898 |
| + captioned bars vouched by the opposite bar | 0.9994 | 0.910 |
| **final clean build, fresh cache** | **0.9994** | **0.909** |

Final build, cold 353 s, warm ~18 s: `-d 6` P 0.9999 R 0.899, `-d 7` P 0.9994 R 0.909 (original scorer 0.9958: 50 lyric_1/2 links), `-d 8` P 0.9975 R 0.916. Seed-split halves at `-d 7`: R 0.917 / 0.917, P 0.9998 / 0.9997 (64 build: 0.827 / 0.864). 13 files (lyric_black and its copies) now fail as all-featureless.

Measured and refused: 256 frame (R 0.871 at `-d 6`, 450 s, 1.5 GB); AREA scaler (same frontier; at 128 for every frame and for interlaced frames only); crop judged on a 64 average; wider rescue reach (lab); chain envelope fill; span credit from every match (31 false links at `-d 7`); featureless floor 25 / 8 (+11 links); zoom on both sides (+92 links, +7 false: screen recordings of one player); inset confirmation of plain matches, strict and as a chance-level veto (cost more than it removed); lenient bar peeling without the opposite bar vouching (ocean_dark, Blender panels: -18 seed copies, +19 false links); tiny rule off (139 false links).

## --keyframe-interval x --min-keyframes on VIDS and VIDS2 (2026-11-21)

Binary 0.41.2 (static, dedf393), `-d 6 -p 20`, cold cache per cell, scored with `SHARED_LYRIC=1`. Harness: `kf_sweep2.py`, `kf_report2.py`; full table `out/kfsweep2/report.txt`, `cells.csv`. Grid: interval 5/13/21/29 x `-m` 0,2,4,6,8,10,12,16,20,24,28 (plus 40/56/80 at 5/13/29); each cell also re-run warm at `-p 0`. Default (interval 0) repeated 6x per corpus: VIDS 147-160 s, VIDS2 266-287 s (5% spread); 26 cells timed twice agree within ~5% bar three outliers (80 vs 101, 79, 103 s).

- Default: VIDS F1 0.947 (154 s, 485 MB), VIDS2 0.918 (275 s, 543 MB). Any interval: RSS 200-410 MB, `-m` does not move it.
- **F1 rises with `-m` all the way to 80** on both corpora; nothing peaks at 12, 20 or 28. At `-m 80` intervals 13 and 29 are within 0.001 F1 of each other (the floor decides for almost every file).
- **The `-m` cost depends on the library**: VIDS (short clips) i29 31 s at m0 -> 56 s at m28 -> 96 s at m80; VIDS2 58 -> 66 -> 95 s. At i5 `-m` costs ~nothing up to 28.
- **"8 and 16 lose pairs" is wrong now.** 16 is never below 12 on either corpus at any interval. The one dip is a single rung at 6 or 8 (after 4): VIDS i21/i29 at m6 (-1 pt recall), i5/i13 at m8 (-0.5/-1 pt); VIDS2 i29 m6 (-3.5 pts), i21 m8 (-1.3), i13 m6 (flat). At `-p 0` recall rises at every one of those rungs: the dip is the `-p` gate diluting coverage, as before, but it sits at 6-8 and is gone by 10.
- **Cost/F1 frontier (sum of both corpora's wall, mean F1)**: every point on it is interval 21-29 with `-m` varied, except the top: i29 m12 96 s F1 0.769; i29 m20 107 s 0.835; i29 m28 123 s 0.861; i29 m40 144 s 0.886; i29 m56 169 s 0.905; i29 m80 190 s 0.914; i13 m80 215 s 0.919; i5 m80 264 s 0.921; default 435 s 0.932. Interval 5 is dominated everywhere below m80. `-m` is the better knob than the interval.
- VIDS2 precision improves with sampling (default 238 false links, 0.9585; i5 ~70, 0.985; i29 m<=20 under 20) because look-alike lecture/match frames are thinned; it falls again at m80 (124).

### `-m 64` and why interval 29 (2026-11-21, same day)

- `-m 64` at interval 5/13/29/60/120: VIDS F1 0.9399/0.9379/0.9377/0.9377/0.9377 at 99/88/80/86/87 s; VIDS2 0.8875/0.8806/0.8726/0.8681/0.8656 at 157/107/79/79/76 s. VIDS2 false links 94 at every interval (61 at m56, 124 at m80).
- 29 is not special. The interval only reaches files longer than interval x `-m`: at m80, 6 VIDS files and 27 VIDS2 files at i29. Every pair that differs between i13 and i29 at m80 has a file over 1,040 s (VIDS2: 87 of 90 only-i13 pairs have one over 2,320 s). Past ~30 s the time saved flattens (VIDS2 79 -> 76 s from 29 to 120) while F1 keeps slipping, so ~30 is the knee of these two corpora, not a constant.
