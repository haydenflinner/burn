//! Visual report of what the simd kernel tests exercise.
//!
//! Run `BURN_SIMD_REPORT=1 cargo test -p burn-ndarray simd::report` to
//! write `simd-test-report.html` in the crate root: every test's inputs
//! and simd outputs rendered as 2-D grids.

use alloc::{format, string::String, vec::Vec};

use burn_backend::{Element, ElementConversion};

use crate::SharedArray;
use crate::testutil::{arr, arr_at, f_order, gapped, nchw, nhwc, pos, vals, vals_b};

use super::avgpool::try_avg_pool2d_simd;
use super::binary::try_binary_simd as binop;
use super::binary_elemwise::{
    VecAdd, VecBitAnd, VecBitOr, VecBitXor, VecClamp, VecDiv, VecMax, VecMin, VecMul, VecSub,
    try_binary_scalar_simd,
};
use super::cmp::{VecEquals, VecGreater, VecGreaterEq, VecLower, VecLowerEq, try_cmp_simd};
use super::conv::try_conv2d_simd;
use super::lanes;
use super::maxpool::try_max_pool2d_simd;
use super::unary::{RecipVec, VecAbs, VecBitNot, try_unary_simd};

/// A labelled tensor to render.
struct Arr {
    label: &'static str,
    shape: Vec<usize>,
    data: Vec<f64>,
    is_bool: bool,
    /// Names for the leading dims when ndim > 2 (e.g. `["n", "c"]`).
    axes: &'static [&'static str],
}

/// One rendered test case.
struct Case {
    /// The `#[test]` functions this case belongs to.
    tests: &'static [&'static str],
    title: String,
    arrays: Vec<Arr>,
    rejected: bool,
    /// Optional pre-rendered HTML block (e.g. a parameter table).
    extra: String,
}

fn num<T: Element + ElementConversion>(a: &SharedArray<T>) -> Arr {
    Arr {
        label: "?",
        shape: a.shape().to_vec(),
        data: a.iter().map(|v| v.elem::<f64>()).collect(),
        is_bool: false,
        axes: &[],
    }
}

fn boolean(a: &SharedArray<bool>) -> Arr {
    Arr {
        label: "?",
        shape: a.shape().to_vec(),
        data: a.iter().map(|v| if *v { 1.0 } else { 0.0 }).collect(),
        is_bool: true,
        axes: &[],
    }
}

impl Arr {
    fn named(mut self, label: &'static str, axes: &'static [&'static str]) -> Self {
        self.label = label;
        self.axes = axes;
        self
    }
}

const NCHW: &[&str] = &["n", "c"];

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();

    // ---- avgpool ---------------------------------------------------------
    let x = nhwc::<f32>(2, 9 * lanes::<f32>() + 1, 6, 5);
    let out = try_avg_pool2d_simd::<f32>(x.clone(), [3, 2], [2, 1], [1, 0], false).unwrap();
    cases.push(Case {
        tests: &["avgpool::matches_scalar_f32"],
        title: "avgpool f32 — kernel [3,2], stride [2,1], pad [1,0], NHWC input".into(),
        arrays: vec![
            num(&x).named("input [n,c,h,w]", NCHW),
            num(&out).named("output [n,c,h,w]", NCHW),
        ],
        rejected: false,
        extra: String::new(),
    });

    let x = nhwc::<f64>(1, 9 * lanes::<f64>() + 1, 5, 4);
    let out = try_avg_pool2d_simd::<f64>(x.clone(), [2, 3], [1, 2], [1, 1], true).unwrap();
    cases.push(Case {
        tests: &["avgpool::matches_scalar_f64_counting_pad"],
        title: "avgpool f64 — kernel [2,3], stride [1,2], pad [1,1], count_include_pad".into(),
        arrays: vec![
            num(&x).named("input [n,c,h,w]", NCHW),
            num(&out).named("output [n,c,h,w]", NCHW),
        ],
        rejected: false,
        extra: String::new(),
    });

    let x = nchw::<f32>(1, 8, 4, 4);
    cases.push(Case {
        tests: &["avgpool::rejects_standard_layout"],
        title: "avgpool f32 — standard NCHW layout is rejected (channel stride ≠ 1)".into(),
        arrays: vec![num(&x).named("input [n,c,h,w]", NCHW)],
        rejected: true,
        extra: String::new(),
    });

    // ---- maxpool ---------------------------------------------------------
    let x = nhwc::<f32>(2, 9 * lanes::<f32>() + 1, 6, 5);
    let out = try_max_pool2d_simd::<f32>(x.clone(), [3, 2], [2, 1], [1, 0], [1, 1]).unwrap();
    cases.push(Case {
        tests: &["maxpool::matches_scalar_f32"],
        title: "maxpool f32 — kernel [3,2], stride [2,1], pad [1,0], dilation [1,1]".into(),
        arrays: vec![
            num(&x).named("input [n,c,h,w]", NCHW),
            num(&out).named("output [n,c,h,w]", NCHW),
        ],
        rejected: false,
        extra: String::new(),
    });

    let x = nhwc::<i32>(1, 9 * lanes::<i32>() + 1, 5, 4);
    let out = try_max_pool2d_simd::<i32>(x.clone(), [2, 2], [1, 1], [1, 1], [1, 1]).unwrap();
    cases.push(Case {
        tests: &["maxpool::matches_naive_i32"],
        title: "maxpool i32 — kernel [2,2], stride [1,1], pad [1,1]".into(),
        arrays: vec![
            num(&x).named("input [n,c,h,w]", NCHW),
            num(&out).named("output [n,c,h,w]", NCHW),
        ],
        rejected: false,
        extra: String::new(),
    });

    let x = nchw::<f32>(1, 8, 4, 4);
    cases.push(Case {
        tests: &["maxpool::rejects_standard_layout"],
        title: "maxpool f32 — standard NCHW layout is rejected".into(),
        arrays: vec![num(&x).named("input [n,c,h,w]", NCHW)],
        rejected: true,
        extra: String::new(),
    });

    // ---- conv ------------------------------------------------------------
    /// `(stride, padding, dilation, groups)` per conv launch arm.
    type ConvMode = ([usize; 2], [usize; 2], [usize; 2], usize);

    let modes: Vec<ConvMode> = vec![
        ([1, 1], [0, 0], [1, 1], 1),
        ([1, 1], [1, 1], [1, 1], 1),
        ([2, 2], [0, 0], [1, 1], 1),
        ([2, 1], [1, 0], [1, 1], 1),
        ([1, 1], [0, 0], [1, 1], 2),
        ([1, 1], [1, 1], [1, 1], 2),
        ([2, 2], [0, 0], [1, 1], 2),
        ([2, 2], [1, 1], [1, 1], 2),
        ([1, 1], [0, 0], [2, 2], 1),
    ];
    let mut extra = String::from(
        "<table class=params><tr><th>stride</th><th>padding</th><th>dilation</th><th>groups</th><th>launch arm</th></tr>",
    );
    for (s, p, d, g) in &modes {
        let arm = match (
            p.iter().any(|v| *v > 0),
            s.iter().any(|v| *v > 1) || d.iter().any(|v| *v > 1),
            *g > 1,
        ) {
            (false, false, false) => "plain",
            (true, false, false) => "padded",
            (false, true, false) => "strided",
            (true, true, false) => "padded + strided",
            (false, false, true) => "grouped",
            (true, false, true) => "grouped + padded",
            (false, true, true) => "grouped + strided",
            (true, true, true) => "grouped + padded + strided",
        };
        extra.push_str(&format!(
            "<tr><td>{s:?}</td><td>{p:?}</td><td>{d:?}</td><td>{g}</td><td>{arm}</td></tr>"
        ));
    }
    extra.push_str("</table>");

    let groups = 1;
    let out_channels = 2 * lanes::<f32>() * groups;
    let x = nchw::<f32>(2, 2 * groups, 7, 6);
    let w = arr_at(&[out_channels, 2, 3, 3], vals(out_channels * 2 * 9));
    let bias = arr(vals(out_channels));
    let out = try_conv2d_simd::<f32>(
        x.clone(),
        w.clone(),
        Some(bias.clone()),
        burn_backend::ops::ConvOptions::new([1, 1], [0, 0], [1, 1], groups),
    )
    .unwrap();
    cases.push(Case {
        tests: &["conv::matches_scalar_for_all_launch_modes"],
        title: "conv2d f32 — one input rendered; every launch arm covered by the parameter matrix"
            .into(),
        arrays: vec![
            num(&x).named("input [n,c,h,w]", NCHW),
            num(&w).named("weights [o,i,kh,kw]", &["o", "i"]),
            num(&bias).named("bias [o]", &[]),
            num(&out).named("output [n,c,h,w]", NCHW),
        ],
        rejected: false,
        extra,
    });

    // ---- elementwise binary ----------------------------------------------
    let lhs = arr(vals::<f32>(97));
    let rhs = arr(vals_b::<f32>(97));
    for (name, out) in [
        (
            "div",
            binop::<f32, f32, f32, f32, VecDiv>(lhs.clone(), arr(pos(97))).unwrap(),
        ),
        (
            "min",
            binop::<f32, f32, f32, f32, VecMin>(lhs.clone(), rhs.clone()).unwrap(),
        ),
        (
            "max",
            binop::<f32, f32, f32, f32, VecMax>(lhs.clone(), rhs.clone()).unwrap(),
        ),
        (
            "add",
            binop::<f32, f32, f32, f32, VecAdd>(lhs.clone(), rhs.clone()).unwrap(),
        ),
        (
            "sub",
            binop::<f32, f32, f32, f32, VecSub>(lhs.clone(), rhs.clone()).unwrap(),
        ),
        (
            "mul",
            binop::<f32, f32, f32, f32, VecMul>(lhs.clone(), rhs.clone()).unwrap(),
        ),
    ] {
        cases.push(Case {
            tests: &["binary::matches_scalar_for_each_op"],
            title: format!("binary f32 — {name}"),
            arrays: vec![
                num(&lhs).named("lhs", &[]),
                if name == "div" {
                    num(&arr::<f32>(pos(97))).named("rhs", &[])
                } else {
                    num(&rhs).named("rhs", &[])
                },
                num(&out).named("out", &[]),
            ],
            rejected: false,
            extra: String::new(),
        });
    }

    let lhs = arr(vals::<i32>(97));
    let rhs = arr(vals_b::<i32>(97));
    for (name, out) in [
        (
            "bitand",
            binop::<i32, i32, i32, i32, VecBitAnd>(lhs.clone(), rhs.clone()).unwrap(),
        ),
        (
            "bitor",
            binop::<i32, i32, i32, i32, VecBitOr>(lhs.clone(), rhs.clone()).unwrap(),
        ),
        (
            "bitxor",
            binop::<i32, i32, i32, i32, VecBitXor>(lhs.clone(), rhs.clone()).unwrap(),
        ),
    ] {
        cases.push(Case {
            tests: &["binary::matches_scalar_for_each_op"],
            title: format!("binary i32 — {name}"),
            arrays: vec![
                num(&lhs).named("lhs", &[]),
                num(&rhs).named("rhs", &[]),
                num(&out).named("out", &[]),
            ],
            rejected: false,
            extra: String::new(),
        });
    }

    cases.push(Case {
        tests: &["binary::rejects_non_standard_layout"],
        title: "binary f32 — Fortran-order rhs is rejected".into(),
        arrays: vec![num(&f_order::<f32>([12, 8], vals(96))).named("rhs (f-order)", &[])],
        rejected: true,
        extra: String::new(),
    });

    // ---- elementwise scalar ops ------------------------------------------
    let input = arr(vals::<f32>(97));
    for (name, out) in [
        (
            "add 2",
            try_binary_scalar_simd::<f32, f32, f32, f32, VecAdd>(input.clone(), 2.0).unwrap(),
        ),
        (
            "mul 2",
            try_binary_scalar_simd::<f32, f32, f32, f32, VecMul>(input.clone(), 2.0).unwrap(),
        ),
        (
            "div 2",
            try_binary_scalar_simd::<f32, f32, f32, f32, VecDiv>(input.clone(), 2.0).unwrap(),
        ),
        (
            "min 3",
            try_binary_scalar_simd::<f32, f32, f32, f32, VecMin>(input.clone(), 3.0).unwrap(),
        ),
        (
            "max 3",
            try_binary_scalar_simd::<f32, f32, f32, f32, VecMax>(input.clone(), 3.0).unwrap(),
        ),
        (
            "clamp [3,7]",
            try_binary_scalar_simd::<f32, f32, f32, f32, VecClamp>(input.clone(), (3.0, 7.0))
                .unwrap(),
        ),
    ] {
        cases.push(Case {
            tests: &["binary_elemwise::matches_scalar_for_each_op"],
            title: format!("scalar-op f32 — {name}"),
            arrays: vec![num(&input).named("input", &[]), num(&out).named("out", &[])],
            rejected: false,
            extra: String::new(),
        });
    }

    cases.push(Case {
        tests: &["binary_elemwise::rejects_non_contiguous"],
        title: "scalar-op f32 — stride-2 gapped input is rejected".into(),
        arrays: vec![num(&gapped::<f32>(vals(96))).named("input (gapped)", &[])],
        rejected: true,
        extra: String::new(),
    });

    // ---- comparisons -----------------------------------------------------
    let lhs = arr(vals::<f32>(97));
    let rhs = arr(vals_b::<f32>(97));
    for (name, out) in [
        (
            "eq",
            try_cmp_simd::<f32, f32, VecEquals>(lhs.clone(), rhs.clone()).unwrap(),
        ),
        (
            "gt",
            try_cmp_simd::<f32, f32, VecGreater>(lhs.clone(), rhs.clone()).unwrap(),
        ),
        (
            "ge",
            try_cmp_simd::<f32, f32, VecGreaterEq>(lhs.clone(), rhs.clone()).unwrap(),
        ),
        (
            "le",
            try_cmp_simd::<f32, f32, VecLowerEq>(lhs.clone(), rhs.clone()).unwrap(),
        ),
        (
            "lt",
            try_cmp_simd::<f32, f32, VecLower>(lhs.clone(), rhs.clone()).unwrap(),
        ),
    ] {
        cases.push(Case {
            tests: &["cmp::matches_scalar_for_each_op"],
            title: format!("cmp f32 — {name}"),
            arrays: vec![
                num(&lhs).named("lhs", &[]),
                num(&rhs).named("rhs", &[]),
                boolean(&out).named("out (bool)", &[]),
            ],
            rejected: false,
            extra: String::new(),
        });
    }

    // ---- unary -----------------------------------------------------------
    let input = arr(vals::<f32>(97));
    let out = try_unary_simd::<f32, f32, f32, f32, VecAbs>(input.clone()).unwrap();
    cases.push(Case {
        tests: &["unary::covers_abs_and_bitnot"],
        title: "unary f32 — abs".into(),
        arrays: vec![num(&input).named("input", &[]), num(&out).named("out", &[])],
        rejected: false,
        extra: String::new(),
    });

    let input_i = arr::<i32>((0..97).map(|i: i32| i * 7).collect());
    let out = try_unary_simd::<i32, i32, i32, i32, VecBitNot>(input_i.clone()).unwrap();
    cases.push(Case {
        tests: &["unary::covers_abs_and_bitnot"],
        title: "unary i32 — bitnot".into(),
        arrays: vec![
            num(&input_i).named("input", &[]),
            num(&out).named("out", &[]),
        ],
        rejected: false,
        extra: String::new(),
    });

    let input_r = arr::<f32>((1..=32).map(|i| i as f32).collect());
    let out = try_unary_simd::<f32, f32, f32, f32, RecipVec>(input_r.clone()).unwrap();
    cases.push(Case {
        tests: &["unary::simd_recip_matches_scalar_division"],
        title: "unary f32 — reciprocal".into(),
        arrays: vec![
            num(&input_r).named("input", &[]),
            num(&out).named("out", &[]),
        ],
        rejected: false,
        extra: String::new(),
    });

    cases.push(Case {
        tests: &["unary::rejects_non_contiguous"],
        title: "unary f32 — stride-2 gapped input is rejected".into(),
        arrays: vec![num(&gapped::<f32>(vals(96))).named("input (gapped)", &[])],
        rejected: true,
        extra: String::new(),
    });

    cases
}

// ---- HTML rendering -------------------------------------------------------

fn fmt_num(v: f64) -> String {
    if v.is_nan() {
        "NaN".into()
    } else if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v:.3}");
        s.trim_end_matches('0').trim_end_matches('.').into()
    }
}

fn cell_color(v: f64, min: f64, range: f64, is_bool: bool) -> String {
    if v.is_nan() {
        return "#e0e0e0".into();
    }
    if is_bool {
        return if v == 0.0 {
            "#e8e8e8".into()
        } else {
            "#a8d5a2".into()
        };
    }
    // Diverging: low → blue tint, high → red tint.
    let t = if range > 0.0 { (v - min) / range } else { 0.5 };
    format!("hsl({:.0}, 65%, 85%)", (1.0 - t) * 220.0)
}

fn render_array(a: &Arr) -> String {
    let (min, max) = a
        .data
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    let range = max - min;

    let mut html = format!(
        "<div class=tensor><div class=tlabel>{} <span class=dims>{:?}</span></div>",
        a.label, a.shape
    );

    let dims = a.shape.len();
    let cell = |i: usize| -> String {
        let v = a.data[i];
        format!(
            "<td class=c style=\"background:{}\">{}</td>",
            cell_color(v, min, range, a.is_bool),
            fmt_num(v)
        )
    };

    let (h, w) = match dims {
        0 => (1, 1),
        1 => (1, a.shape[0]),
        _ => (a.shape[dims - 2], a.shape[dims - 1]),
    };
    let sheets: usize = if dims <= 2 {
        1
    } else {
        a.shape[..dims - 2].iter().product()
    };

    html.push_str("<div class=sheets>");
    for sheet in 0..sheets {
        if sheets > 1 {
            // Decompose the sheet index into leading-dim coordinates.
            let mut idx = sheet;
            let mut coords = Vec::new();
            for d in (0..dims - 2).rev() {
                coords.insert(0, idx % a.shape[d]);
                idx /= a.shape[d];
            }
            let label = a
                .axes
                .iter()
                .zip(&coords)
                .map(|(name, i)| format!("{name}={i}"))
                .collect::<Vec<_>>()
                .join(" ");
            html.push_str(&format!("<div class=sheet><div class=slabel>{label}</div>"));
        } else {
            html.push_str("<div class=sheet>");
        }
        html.push_str("<table class=grid>");
        for r in 0..h {
            html.push_str("<tr>");
            for c in 0..w {
                html.push_str(&cell(sheet * h * w + r * w + c));
            }
            html.push_str("</tr>");
        }
        html.push_str("</table></div>");
    }
    html.push_str("</div></div>");
    html
}

fn render(cases: &[Case]) -> String {
    let mut html = String::from(
        r#"<!doctype html><meta charset=utf-8>
<title>burn-ndarray simd test report</title>
<style>
body{font:13px/1.4 -apple-system,sans-serif;margin:24px;color:#222}
h1{font-size:18px} h2{font-size:15px;margin:28px 0 6px}
.case{border:1px solid #ddd;border-radius:6px;padding:10px 14px;margin:10px 0}
.tests span{font:10px monospace;background:#f0f0f0;border-radius:3px;padding:1px 5px;margin-right:4px}
.rej{color:#a33;font-weight:600;font-size:11px}
.badge{font-size:10px;color:#666}
.tensor{margin:8px 12px 8px 0;display:inline-block;vertical-align:top}
.tlabel{font:11px monospace;color:#444;margin-bottom:2px}
.dims{color:#999}
.sheets{display:flex;flex-wrap:wrap;gap:6px}
.sheet{display:inline-block}
.slabel{font:9px monospace;color:#888}
table.grid{border-collapse:collapse}
table.grid td{border:1px solid #bbb;font:8px monospace;width:22px;height:15px;text-align:center;padding:0;overflow:hidden}
table.params{border-collapse:collapse;margin:8px 0}
table.params td,table.params th{border:1px solid #ccc;padding:2px 10px;font:11px monospace;text-align:left}
</style>
<h1>burn-ndarray simd test report</h1>
<p class=badge>Each case shows the tensors fed to (and produced by) the simd kernels in the unit tests.
Cells are colored by value within each tensor; green = true, gray = false for bool outputs.
"rejected" cases are inputs the simd path declines so the scalar fallback runs.</p>
"#,
    );
    for case in cases {
        html.push_str("<div class=case><div class=tests>");
        for t in case.tests {
            html.push_str(&format!("<span>{t}</span>"));
        }
        if case.rejected {
            html.push_str("<span class=rej>REJECTED → scalar fallback</span>");
        }
        html.push_str(&format!("</div><h2>{}</h2>", case.title));
        html.push_str(&case.extra);
        for a in &case.arrays {
            html.push_str(&render_array(a));
        }
        html.push_str("</div>");
    }
    html
}

#[test]
fn write_report() {
    let path: String = match std::env::var("BURN_SIMD_REPORT").as_deref() {
        Err(_) => return,
        // Presence enables the report; a path-like value overrides the output.
        Ok(v) if v.is_empty() || v == "1" || v.eq_ignore_ascii_case("true") => {
            "simd-test-report.html".into()
        }
        Ok(v) => v.into(),
    };
    std::fs::write(&path, render(&cases())).unwrap();
}
