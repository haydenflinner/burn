use super::*;
use burn_core::tensor::{DType, Distribution, Element, Tolerance, s};
use burn_linalg::qr;

const REL: f32 = 5e-3;
// Householder updates are expressed as matmuls, for which accelerated CUDA
// kernels can use reduced-precision inputs and accumulate errors around 1e-3.
const ABS: f32 = 2e-3;

// ---------------------------------------------------------------------
// Small Matrices
// ---------------------------------------------------------------------

#[test]
fn test_qr_1x1() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[5.0]], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_2x2() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[4.0, 3.0], [6.0, 3.0]], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_3x3() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[4.0, 7.0, 3.0], [6.0, 1.0, 3.0], [8.0, 3.0, 7.0]], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_identity_matrix() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_zero_matrix() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_singular_maxtrix() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(
        [
            [0.0, 4.0, 2.0, 6.0],
            [0.0, 2.0, 2.0, 11.0],
            [0.0, 3.0, 9.0, 6.0],
            [0.0, 7.0, 10.0, 9.0],
        ],
        &device,
    );
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_identity_singular_maxtrix() {
    let device = Default::default();
    let tensor = TestTensor::<2>::ones([4, 4], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_v0_zero() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[0.0, 4.0, 2.0], [1.0, 2.0, 2.0], [2.0, 3.0, 9.0]], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_diagonilized_col() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[1.0, 4.0, 2.0], [0.0, 2.0, 2.0], [0.0, 3.0, 9.0]], &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_5x5_reduced_shape() {
    let device = Default::default();
    let tensor = TestTensor::<2>::ones([5, 5], &device);
    let (q, r) = qr::<2>(tensor.clone(), true);
    assert_eq!(q.dims(), [5, 5]);
    assert_eq!(r.dims(), [5, 5]);
}

#[test]
fn test_qr_10x5_reduced_shape() {
    let device = Default::default();
    let tensor = TestTensor::<2>::ones([10, 5], &device);
    let (q, r) = qr::<2>(tensor.clone(), true);
    assert_eq!(q.dims(), [10, 5]);
    assert_eq!(r.dims(), [5, 5]);
}

#[test]
fn test_qr_5x10_reduced_shape() {
    let device = Default::default();
    let tensor = TestTensor::<2>::ones([5, 10], &device);
    let (q, r) = qr::<2>(tensor.clone(), true);
    assert_eq!(q.dims(), [5, 5]);
    assert_eq!(r.dims(), [5, 10]);
}

#[test]
fn test_qr_2d_square() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([6, 6], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_2d_square_reduced() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([6, 6], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), true);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_2d_tall() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([8, 5], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_2d_tall_reduced() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([8, 5], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_singular_2d_tall() {
    let device = Default::default();
    let zeros = TestTensor::<2>::zeros([8, 1], &device);
    let mut tensor = TestTensor::<2>::random([8, 5], Distribution::Default, &device);
    tensor = tensor.slice_assign(s![.., 0..1], zeros);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_singular_2d_tall_reduces() {
    let device = Default::default();
    let zeros = TestTensor::<2>::zeros([8, 1], &device);
    let mut tensor = TestTensor::<2>::random([8, 5], Distribution::Default, &device);
    tensor = tensor.slice_assign(s![.., 0..1], zeros);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_2d_wide() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([5, 8], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_2d_wide_reduced() {
    let device = Default::default();
    let tensor = TestTensor::<2>::random([5, 8], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), true);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_singular_2d_wide() {
    let device = Default::default();
    let zeros = TestTensor::<2>::zeros([5, 1], &device);
    let mut tensor = TestTensor::<2>::random([5, 8], Distribution::Default, &device);
    tensor = tensor.slice_assign(s![.., 0..1], zeros);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_singular_2d_wide_reduced() {
    let device = Default::default();
    let zeros = TestTensor::<2>::zeros([5, 1], &device);
    let mut tensor = TestTensor::<2>::random([5, 8], Distribution::Default, &device);
    tensor = tensor.slice_assign(s![.., 0..1], zeros);
    let (q, r) = qr::<2>(tensor.clone(), true);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_medium_tall() {
    if !matches!(FloatElem::dtype(), DType::F32) {
        return; // skip on lower precision dtypes (e.g. f16)
    }
    let device = Default::default();
    let tensor = TestTensor::<2>::random([256, 128], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_medium_wide() {
    if !matches!(FloatElem::dtype(), DType::F32) {
        return; // skip on lower precision dtypes (e.g. f16)
    }
    let device = Default::default();
    let tensor = TestTensor::<2>::random([128, 256], Distribution::Default, &device);
    let (q, r) = qr::<2>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

// ---------------------------------------------------------------------
// 3D Tensors (1 batch dimension)
// ---------------------------------------------------------------------

#[test]
fn test_qr_3d_square() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([3, 6, 6], Distribution::Default, &device);
    let (q, r) = qr::<3>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_3d_tall() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([3, 8, 5], Distribution::Default, &device);
    let (q, r) = qr::<3>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_3d_wide() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([3, 5, 8], Distribution::Default, &device);
    let (q, r) = qr::<3>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_3d_8x6_reduced_shape() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([3, 8, 6], Distribution::Default, &device);
    let (q, r) = qr::<3>(tensor.clone(), true);
    assert_eq!(q.dims(), [3, 8, 6]);
    assert_eq!(r.dims(), [3, 6, 6]);
}

// ---------------------------------------------------------------------
// 4D Tensors (2 batch dimensions)
// ---------------------------------------------------------------------

#[test]
fn test_qr_4d_square() {
    let device = Default::default();
    let tensor = TestTensor::<4>::random([2, 2, 6, 6], Distribution::Default, &device);
    let (q, r) = qr::<4>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_4d_square_with_singularities() {
    let device = Default::default();
    let ones = TestTensor::<4>::ones([1, 1, 6, 6], &device);
    let mut tensor = TestTensor::<4>::random([2, 2, 6, 6], Distribution::Default, &device);
    tensor = tensor.slice_assign(s![0..1, 0..1, .., ..], ones.clone());
    tensor = tensor.slice_assign(s![2..3, 1..2, .., ..], ones.clone());
    let (q, r) = qr::<4>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_4d_tall() {
    let device = Default::default();
    let tensor = TestTensor::<4>::random([2, 2, 8, 5], Distribution::Default, &device);
    let (q, r) = qr::<4>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_4d_tall_with_singularities() {
    let device = Default::default();
    let ones = TestTensor::<4>::ones([1, 1, 8, 5], &device);
    let mut tensor = TestTensor::<4>::random([2, 2, 8, 5], Distribution::Default, &device);
    tensor = tensor.slice_assign(s![0..1, 0..1, .., ..], ones.clone());
    tensor = tensor.slice_assign(s![2..3, 1..2, .., ..], ones.clone());
    let (q, r) = qr::<4>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_4d_wide() {
    let device = Default::default();
    let tensor = TestTensor::<4>::random([2, 2, 5, 8], Distribution::Default, &device);
    let (q, r) = qr::<4>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_4d_wide_with_singularities() {
    let device = Default::default();
    let ones = TestTensor::<4>::ones([1, 1, 5, 8], &device);
    let mut tensor = TestTensor::<4>::random([2, 2, 5, 8], Distribution::Default, &device);
    tensor = tensor.slice_assign(s![0..1, 0..1, .., ..], ones.clone());
    tensor = tensor.slice_assign(s![2..3, 1..2, .., ..], ones.clone());
    let (q, r) = qr::<4>(tensor.clone(), false);
    let qr = q.matmul(r);
    let tolerance = Tolerance::rel_abs(REL, ABS).set_half_precision_absolute(5e-2);
    qr.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), tolerance);
}

#[test]
fn test_qr_4d_12x4_reduced_shape() {
    let device = Default::default();
    let tensor = TestTensor::<4>::random([2, 3, 12, 4], Distribution::Default, &device);
    let (q, r) = qr::<4>(tensor.clone(), true);
    assert_eq!(q.dims(), [2, 3, 12, 4]);
    assert_eq!(r.dims(), [2, 3, 4, 4]);
}

/// Regression: a matvec on a strided (sliced, non-zero offset) lhs must not
/// read misaligned lines — cubecl-matmul `gemm`/`gemv` kernels picked a
/// vector size that didn't divide the row stride and silently loaded the
/// wrong elements on odd rows.
#[test]
fn test_matvec_strided_lhs() {
    let device = Default::default();
    // a[i][j] = i*10 + j — every element identifiable
    let a = TestTensor::<2>::from_data(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
            [40.0, 41.0, 42.0, 43.0, 44.0, 45.0],
            [50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
        ],
        &device,
    );
    let w = TestTensor::<2>::from_data([[1.0], [2.0], [3.0], [4.0]], &device);
    // strided 6x4 view: cols 2..6 of a
    let s = a.clone().slice(s![.., 2..6]);
    let got = s.matmul(w.clone());
    // row i: (10i+2)*1 + (10i+3)*2 + (10i+4)*3 + (10i+5)*4 = 100i+40
    let want = TestTensor::<2>::from_data(
        [[40.0], [140.0], [240.0], [340.0], [440.0], [540.0]],
        &device,
    );
    got.into_data()
        .assert_approx_eq::<FloatElem>(&want.into_data(), Tolerance::default());

    // transposed-view lhs: (4x6)^T -> 6x4, strides (1, 6)
    let st = a.slice(s![0..4, ..]).transpose();
    let got = st.matmul(w);
    // row i: sum_j (10j+i)*(j+1) = 200 + 10i
    let want = TestTensor::<2>::from_data(
        [[200.0], [210.0], [220.0], [230.0], [240.0], [250.0]],
        &device,
    );
    got.into_data()
        .assert_approx_eq::<FloatElem>(&want.into_data(), Tolerance::default());
}

// ---------------------------------------------------------------------
// Tensor Check Panics
// ---------------------------------------------------------------------

#[test]
#[should_panic]
fn test_qr_panic_rank_less_than_2() {
    // Fails check: D >= 2
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([1.0, 2.0, 3.0], &device);
    let _ = qr::<1>(tensor, false);
}
