//! Depthwise (`groups == in_channels`) weight gradient against the reference
//! backend, repeated: the CubeCL fast path folds the batch into the channels
//! (`conv_weight_grad_depthwise`) and was reported to be nondeterministically
//! wrong on the CPU runtime. `DW_ITERS=n` repeats each case n times in-process.
//!
//! Root cause (found with these tests): the fold is `reshape(permute(input))`,
//! a reshape of a permuted view that keeps the rank. `reshape` falls back to
//! `cubecl_std::tensor::copy_into`, which on a CPU client with a non-unit last
//! input stride and equal *rank* picks the "perpendicular" (transposing) copy.
//! That kernel indexes the input with the output's coordinates, so it is only
//! valid when input and output have the *same shape*: the copy scrambles the
//! data and leaves part of the fresh output unwritten (zeros on a fresh heap,
//! stale bytes later: hence "nondeterministic"). `folded_image_permute_reshape`
//! isolates it without any convolution. Fixed in cubecl by checking
//! `input.shape == output.shape` instead of the rank (cubecl `main` has it; the
//! rev Burn pins, 5e521ef, and d0052727 do not).
use super::*;
use burn_tensor::Tolerance;
use burn_tensor::ops::ConvOptions;
use burn_tensor::{Distribution, module};

fn iters() -> usize {
    std::env::var("DW_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

/// Returns the number of runs (out of `iters()`) whose gradient mismatched.
fn run_case(
    batch: usize,
    channels: usize,
    mult: usize,
    hw: [usize; 2],
    k: [usize; 2],
    options: ConvOptions<2>,
) -> usize {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();
    let mut bad = 0;
    for it in 0..iters() {
        let input = TestTensor::<4>::random(
            [batch, channels, hw[0], hw[1]],
            Distribution::Default,
            &device,
        );
        let weight = TestTensor::<4>::random(
            [channels * mult, 1, k[0], k[1]],
            Distribution::Default,
            &device,
        );
        let out_shape = module::conv2d(input.clone(), weight.clone(), None, options.clone()).dims();
        let output_grad = TestTensor::<4>::random(out_shape, Distribution::Default, &device);

        let input_ref = TestTensor::<4>::from_data(input.to_data(), &ref_device);
        let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
        let og_ref = TestTensor::<4>::from_data(output_grad.to_data(), &ref_device);

        let g = module::conv2d_weight_backward(input, weight, output_grad, options.clone());
        let g_ref = module::conv2d_weight_backward(input_ref, weight_ref, og_ref, options.clone());
        let (g, g_ref) = (g.into_data(), g_ref.into_data());
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            g.assert_approx_eq::<FloatElem>(&g_ref, Tolerance::default());
        }))
        .is_ok();
        if !ok {
            bad += 1;
            eprintln!("DW_MISMATCH it={it} b={batch} c={channels} m={mult} hw={hw:?} k={k:?}");
        }
    }
    bad
}

#[test]
fn depthwise_weight_grad_matches_reference_repeated() {
    let mut bad = 0;
    // (batch, channels, multiplier, hw, kernel, options)
    bad += run_case(2, 2, 1, [4, 4], [3, 3], ConvOptions::new([1, 1], [1, 1], [1, 1], 2));
    bad += run_case(2, 2, 2, [4, 4], [3, 3], ConvOptions::new([1, 1], [0, 0], [1, 1], 2));
    bad += run_case(4, 16, 1, [8, 8], [3, 3], ConvOptions::new([1, 1], [1, 1], [1, 1], 16));
    bad += run_case(4, 32, 1, [16, 16], [3, 3], ConvOptions::new([2, 2], [1, 1], [1, 1], 32));
    bad += run_case(8, 64, 1, [14, 14], [5, 5], ConvOptions::new([1, 1], [2, 2], [1, 1], 64));
    bad += run_case(3, 24, 2, [9, 11], [3, 3], ConvOptions::new([1, 2], [1, 1], [1, 1], 24));
    bad += run_case(1, 96, 1, [28, 28], [3, 3], ConvOptions::new([1, 1], [1, 1], [1, 1], 96));
    bad += run_case(16, 8, 1, [32, 32], [3, 3], ConvOptions::new([1, 1], [1, 1], [1, 1], 8));
    assert_eq!(bad, 0, "{bad} depthwise weight-gradient mismatches");
}

/// Step 1 of `conv_weight_grad_depthwise`: `[N, H, W, C]` -> permute -> reshape
/// `[1, H, W, C * N]` (a reshape of a permuted view, which cannot stay a view).
#[test]
fn folded_image_permute_reshape_matches_reference() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();
    let mut bad = 0;
    for it in 0..iters() {
        for &(n, h, w, c) in &[(2usize, 4usize, 4usize, 2usize), (4, 8, 8, 16), (3, 5, 7, 6)] {
            let x = TestTensor::<4>::random([n, h, w, c], Distribution::Default, &device);
            let x_ref = TestTensor::<4>::from_data(x.to_data(), &ref_device);
            let got = x.clone().permute([1, 2, 3, 0]).reshape([1, h, w, c * n]).into_data();
            let want_ref = x_ref.permute([1, 2, 3, 0]).reshape([1, h, w, c * n]).into_data();
            // Hand-computed: out[0, y, x, ch * n_total + b] = in[b, y, x, ch].
            let src = x.into_data().to_vec::<f32>().unwrap();
            let mut manual = vec![0f32; n * h * w * c];
            for b in 0..n {
                for y in 0..h {
                    for xx in 0..w {
                        for ch in 0..c {
                            manual[((y * w) + xx) * (c * n) + ch * n + b] =
                                src[((b * h + y) * w + xx) * c + ch];
                        }
                    }
                }
            }
            let got_v = got.to_vec::<f32>().unwrap();
            let ref_v = want_ref.to_vec::<f32>().unwrap();
            let n_bad = got_v.iter().zip(&manual).filter(|(a, b)| a != b).count();
            let n_bad_ref = ref_v.iter().zip(&manual).filter(|(a, b)| a != b).count();
            if n_bad != 0 || n_bad_ref != 0 {
                if n * h * w * c <= 64 {
                    eprintln!("  src    = {src:?}\n  want   = {manual:?}\n  cube   = {got_v:?}");
                }
                bad += 1;
                eprintln!(
                    "FOLD_MISMATCH it={it} n={n} h={h} w={w} c={c}: cube wrong in {n_bad}/{} elems, ndarray wrong in {n_bad_ref}",
                    manual.len()
                );
            }
        }
    }
    assert_eq!(bad, 0);
}

/// The folded convolution itself, through the public forward op: image
/// `[1, C*N, H, W]`, filters `[C_out, N, oh, ow]`, `groups = C`.
#[test]
fn folded_grouped_forward_matches_reference() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();
    let mut bad = 0;
    for it in 0..iters() {
        for &(n, c, h, w, pad) in &[
            (2usize, 2usize, 4usize, 4usize, 1usize),
            (4, 8, 8, 8, 1),
            (3, 6, 5, 7, 0),
        ] {
            let (oh, ow) = (h + 2 * pad - 2, w + 2 * pad - 2); // 3x3 "kernel" output
            let image = TestTensor::<4>::random([1, c * n, h, w], Distribution::Default, &device);
            let filt = TestTensor::<4>::random([c, n, oh, ow], Distribution::Default, &device);
            let image_ref = TestTensor::<4>::from_data(image.to_data(), &ref_device);
            let filt_ref = TestTensor::<4>::from_data(filt.to_data(), &ref_device);
            let opts = ConvOptions::new([1, 1], [pad, pad], [1, 1], c);
            let got = module::conv2d(image, filt, None, opts.clone()).into_data();
            let want = module::conv2d(image_ref, filt_ref, None, opts).into_data();
            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                got.assert_approx_eq::<FloatElem>(&want, Tolerance::default());
            }))
            .is_ok();
            if !ok {
                bad += 1;
                eprintln!("FOLDCONV_MISMATCH it={it} n={n} c={c} h={h} w={w} pad={pad}");
            }
        }
    }
    assert_eq!(bad, 0);
}
