# COMP 590 — Assignment 1: Lossless Video Compression

## Approach

I improved the baseline compressor with three changes, all in `main.rs`:

**1. Better prediction (Temporal MED)**

Instead of just using the same pixel from the last frame as the prediction, I look at how the left, top, and top-left neighbors changed between frames. I take the MED (median edge detector) of those three temporal differences and add it to the prior pixel value. This basically says "this pixel probably changed in a similar way to its neighbors." MED is nice because it picks the right neighbor to follow at edges instead of blurring across them.

**2. Multiple contexts (64 total)**

I use 64 arithmetic coding contexts instead of 1. The context is chosen based on how much the left and top neighbors changed (8 bins each, so 8×8 = 64). The idea is that pixels in still areas should have a different probability distribution than pixels in areas with lots of motion.

**3. Bias correction**

For each context, I track the average signed prediction error and use it to adjust future predictions. This fixes cases where the predictor consistently over- or under-shoots in certain regions.

## Results

### bourne.mp4 (1920×1080, 10 frames)

| Method | Compression Ratio |
|--------|-------------------|
| Baseline (1 context, prior pixel) | 2.37x |
| My approach | **8.33x** |

Verified lossless with `-check_decode` — all 10 frames decoded correctly.