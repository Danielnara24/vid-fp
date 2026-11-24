//! Each decoded picture's luma, scaled to the `SIDE` x `SIDE` working frame by
//! code this crate owns, so the result does not depend on the processor.
//!
//! The scaling used to be libswscale's `FAST_BILINEAR` to `GRAY8`, and its
//! output differs by architecture. The cause is not SIMD rounding. On x86 with
//! MMX, `sws_init_context` replaces the horizontal step
//! `((srcW << 16) + dstW / 2) / dstW` with `((srcW - 2) << 16) / (dstW - 2) - 20`
//! and uses a filter-based horizontal scaler, while every other processor runs
//! `ff_hyscale_fast_c` from x = 0. The two sample different source columns, so
//! an x86 and an aarch64 build hashed different pictures of one frame.
//!
//! This is a port of the C path (libswscale 8.1, `FAST_BILINEAR`, luma into
//! full-range `GRAY8`, cpu flags 0), so it is bit-identical to what aarch64
//! builds always produced from 8-bit sources:
//!
//! * horizontal: `ff_hyscale_fast_c`, 2-tap linear interpolation into 15 bits,
//!   the last columns pinned to the last source pixel;
//! * range: `lumRangeToJpeg_c` when the source is limited range (anything but
//!   `yuvj*` and gray), since `GRAY8` is full range;
//! * vertical: `initFilter`'s fast-bilinear branch (2 taps, 12-bit
//!   coefficients, the same reduction, border fix and error-diffused
//!   normalisation), then `yuv2planeX_8_c` / `yuv2plane1_8_c` with the
//!   constant dither swscale uses for 8-bit sources.
//!
//! Planar sources of 9 to 16 bits take swscale's other C route, which this
//! ports too: the same filter built at 14 bits for the horizontal pass
//! (`hScale16To15_c`) and the 8x8 ordered dither on output. Packed, RGB,
//! paletted and semi-planar deep formats (P010), and frames under 8 pixels
//! wide, still go through libswscale (`Fallback`), which is arch-dependent;
//! software decoders rarely produce them for keyframes.

use ffmpeg_next::format::Pixel;
use ffmpeg_next::frame::Video as Frame;

use super::SIDE;

const DST: usize = SIDE;

/// Where a frame's luma is and how to read it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Layout {
    /// Bits per sample: 8, or 9 to 16 stored little-endian in two bytes.
    depth: u32,
    full_range: bool,
}

fn layout(format: Pixel) -> Option<Layout> {
    use Pixel::*;
    let eight = |full_range: bool| Some(Layout { depth: 8, full_range });
    let deep = |depth: u32, full_range: bool| Some(Layout { depth, full_range });
    match format {
        YUV420P | YUV422P | YUV444P | YUV410P | YUV411P | YUV440P | NV12 | NV21 | NV16 | NV24 | NV42
        | YUVA420P | YUVA422P | YUVA444P => eight(false),
        YUVJ420P | YUVJ422P | YUVJ444P | YUVJ440P | YUVJ411P | GRAY8 => eight(true),
        YUV420P9LE | YUV422P9LE | YUV444P9LE => deep(9, false),
        YUV420P10LE | YUV422P10LE | YUV444P10LE | YUV440P10LE | YUVA420P10LE | YUVA422P10LE
        | YUVA444P10LE => deep(10, false),
        YUV420P12LE | YUV422P12LE | YUV444P12LE | YUV440P12LE => deep(12, false),
        YUV420P14LE | YUV422P14LE | YUV444P14LE => deep(14, false),
        YUV420P16LE | YUV422P16LE | YUV444P16LE => deep(16, false),
        GRAY10LE => deep(10, true),
        GRAY12LE => deep(12, true),
        GRAY16LE => deep(16, true),
        _ => Option::None,
    }
}

/// A scaling filter: for each output sample, the first source sample it reads
/// and its taps in fixed point (`initFilter`, fast-bilinear branch).
#[derive(Debug, PartialEq, Eq)]
struct VFilter {
    size: usize,
    pos: Vec<i32>,
    coef: Vec<i16>,
}

fn av_log2(v: u32) -> u32 {
    if v == 0 { 0 } else { 31 - v.leading_zeros() }
}

/// libswscale's `ff_dither_8x8_128`, rows 0-7.
const DITHER_8X8_128: [[u8; 8]; 8] = [
    [36, 68, 60, 92, 34, 66, 58, 90],
    [100, 4, 124, 28, 98, 2, 122, 26],
    [52, 84, 44, 76, 50, 82, 42, 74],
    [116, 20, 108, 12, 114, 18, 106, 10],
    [32, 64, 56, 88, 38, 70, 62, 94],
    [96, 0, 120, 24, 102, 6, 126, 30],
    [48, 80, 40, 72, 54, 86, 46, 78],
    [112, 16, 104, 8, 118, 22, 110, 14],
];

/// `SWS_MAX_REDUCE_CUTOFF`.
const MAX_REDUCE_CUTOFF: f64 = 0.002;

fn vertical_filter(src_h: usize) -> VFilter {
    init_filter(src_h, 1 << 12)
}

/// `initFilter`'s fast-bilinear branch from `src` samples to `DST`, with
/// `one` the fixed-point unit (1 << 12 vertically, 1 << 14 horizontally),
/// on the C path (`filterAlign` 1, no source or destination filter).
fn init_filter(src: usize, one: i64) -> VFilter {
    let (src_h, dst_h) = (src as i64, DST as i64);
    let x_inc = ((src_h << 16) + (dst_h >> 1)) / dst_h;
    let fone: i64 = 1 << (54 - av_log2((src_h / dst_h) as u32).min(8));
    // get_local_pos(c, 0, 0, 1) for both source and destination.
    let pos_ = 128i64;
    let fsize = 2usize;

    let mut filter = vec![0i64; DST * fsize];
    let mut fpos = vec![0i32; DST];
    let mut x_dst_in_src = ((pos_ * x_inc) >> 8) - ((pos_ * 0x8000) >> 7);
    for i in 0..DST {
        let first = (x_dst_in_src - (((fsize as i64) - 1) << 15) + (1 << 15)) >> 16;
        fpos[i] = first as i32;
        for j in 0..fsize {
            let xx = first + j as i64;
            let c = fone - (xx * (1 << 16) - x_dst_in_src).abs() * (fone >> 16);
            filter[i * fsize + j] = c.max(0);
        }
        x_dst_in_src += x_inc;
    }

    // Step 1: shift near-zero taps off the left, count them off the right.
    let cutoff_limit = MAX_REDUCE_CUTOFF * fone as f64;
    let mut min_size = 0usize;
    for i in (0..DST).rev() {
        let mut min = fsize;
        let mut cut: i64 = 0;
        for _ in 0..fsize {
            cut += filter[i * fsize].abs();
            if cut as f64 > cutoff_limit {
                break;
            }
            if i < DST - 1 && fpos[i] >= fpos[i + 1] {
                break;
            }
            for k in 1..fsize {
                filter[i * fsize + k - 1] = filter[i * fsize + k];
            }
            filter[i * fsize + fsize - 1] = 0;
            fpos[i] += 1;
        }
        cut = 0;
        for j in (1..fsize).rev() {
            cut += filter[i * fsize + j].abs();
            if cut as f64 > cutoff_limit {
                break;
            }
            min -= 1;
        }
        min_size = min_size.max(min);
    }
    // filterAlign is 1 on the C path.
    let size = min_size;
    let mut out = vec![0i64; DST * size];
    for i in 0..DST {
        for j in 0..size {
            out[i * size + j] = filter[i * fsize + j];
        }
    }

    // Fix borders.
    let src_w = src_h as i32;
    for i in 0..DST {
        let f = &mut out[i * size..(i + 1) * size];
        if fpos[i] < 0 {
            for j in 1..size {
                let left = (j as i32 + fpos[i]).max(0) as usize;
                f[left] += f[j];
                f[j] = 0;
            }
            fpos[i] = 0;
        }
        if fpos[i] + size as i32 > src_w {
            let shift = fpos[i] + (size as i32 - src_w).min(0);
            let mut acc = 0;
            for j in (0..size).rev() {
                if fpos[i] + j as i32 >= src_w {
                    acc += f[j];
                    f[j] = 0;
                }
            }
            for j in (0..size).rev() {
                if (j as i32) < shift {
                    f[j] = 0;
                } else {
                    f[j] = f[j - shift as usize];
                }
            }
            fpos[i] -= shift;
            f[(src_w - 1 - fpos[i]) as usize] += acc;
        }
    }

    // Normalise to `one`, carrying the rounding error along the taps.
    let rounded_div = |a: i64, b: i64| if a >= 0 { (a + (b >> 1)) / b } else { (a - (b >> 1)) / b };
    let mut coef = vec![0i16; DST * size];
    for i in 0..DST {
        let f = &out[i * size..(i + 1) * size];
        let mut sum: i64 = f.iter().sum();
        sum = (sum + one / 2) / one;
        if sum == 0 {
            sum = 1;
        }
        let mut error = 0;
        for j in 0..size {
            let v = f[j] + error;
            let int_v = rounded_div(v, sum);
            coef[i * size + j] = int_v as i16;
            error = v - int_v * sum;
        }
    }
    VFilter { size, pos: fpos, coef }
}

/// `lumRangeToJpeg_c`'s coefficient and offset for 8-bit output
/// (`solve_range_convert` with libswscale's limited-range luma bounds).
fn range_to_full() -> (i32, i32) {
    let (src_min, src_max, dst_min, dst_max) = (16u64, 235u64, 0u64, 255u64);
    let (src_shift, mult_shift) = (7u32, 14u32);
    let total = mult_shift + src_shift;
    let q = ((dst_max - dst_min) << total) / (src_max - src_min);
    let coeff = (q + (1 << src_shift) - 1) >> src_shift;
    let offset = ((dst_max as i64) << total) - ((src_max as i64) << src_shift) * coeff as i64
        + (1i64 << (mult_shift - 1));
    (coeff as u16 as i32, offset as i32)
}

/// One source size's plan.
struct Plan {
    w: usize,
    h: usize,
    layout: Layout,
    x_inc: u32,
    /// Output columns from here on read the last source pixel alone.
    tail_from: usize,
    /// Deep sources: `hScale16To15_c` over this filter instead of the fast path.
    hfilter: Option<VFilter>,
    v: VFilter,
}

impl Plan {
    fn new(w: usize, h: usize, layout: Layout) -> Plan {
        let x_inc = (((w as u64) << 16) + (DST as u64 >> 1)) / DST as u64;
        let x_inc = x_inc as u32;
        let mut tail_from = DST;
        while tail_from > 0 && (((tail_from - 1) as u64 * x_inc as u64) >> 16) >= (w as u64 - 1) {
            tail_from -= 1;
        }
        let hfilter = (layout.depth > 8).then(|| init_filter(w, 1 << 14));
        Plan { w, h, layout, x_inc, tail_from, hfilter, v: vertical_filter(h) }
    }
}

/// libswscale for what this module does not read itself.
struct Fallback {
    ctx: ffmpeg_next::software::scaling::context::Context,
    out: Frame,
}

/// Scales decoded frames to the working frame; rebuilt per source shape.
pub(super) struct Downscaler {
    plan: Option<Plan>,
    fallback: Option<Fallback>,
    /// Two horizontally scaled rows (15-bit), and which source rows they hold.
    lines: [[i16; DST]; 2],
    line_rows: [i64; 2],
    range: Option<(i32, i32)>,
}

impl Default for Downscaler {
    fn default() -> Self {
        Downscaler { plan: None, fallback: None, lines: [[0; DST]; 2], line_rows: [-1, -1], range: None }
    }
}

impl Downscaler {
    /// Scale `frame` into `out` (`SIDE * SIDE` bytes, row-major).
    pub(super) fn run(&mut self, frame: &Frame, out: &mut [u8]) -> Result<(), ffmpeg_next::Error> {
        let (w, h) = (frame.width() as usize, frame.height() as usize);
        match layout(frame.format()) {
            Some(l) if w >= 8 && h >= 1 => {
                self.fallback = None;
                let fresh = self.plan.as_ref().is_none_or(|p| p.w != w || p.h != h || p.layout != l);
                if fresh {
                    self.plan = Some(Plan::new(w, h, l));
                    self.range = (!l.full_range).then(range_to_full);
                }
                self.scale_luma(frame, out);
                Ok(())
            }
            _ => self.run_fallback(frame, out),
        }
    }

    fn run_fallback(&mut self, frame: &Frame, out: &mut [u8]) -> Result<(), ffmpeg_next::Error> {
        self.plan = None;
        let rebuild = self.fallback.as_ref().is_none_or(|f| {
            let input = f.ctx.input();
            input.format != frame.format() || input.width != frame.width() || input.height != frame.height()
        });
        if rebuild {
            let ctx = ffmpeg_next::software::scaling::context::Context::get(
                frame.format(),
                frame.width(),
                frame.height(),
                Pixel::GRAY8,
                DST as u32,
                DST as u32,
                ffmpeg_next::software::scaling::flag::Flags::FAST_BILINEAR,
            )?;
            self.fallback = Some(Fallback { ctx, out: Frame::empty() });
        }
        let f = self.fallback.as_mut().expect("built above");
        f.ctx.run(frame, &mut f.out)?;
        let (data, stride) = (f.out.data(0), f.out.stride(0));
        for y in 0..DST {
            out[y * DST..(y + 1) * DST].copy_from_slice(&data[y * stride..y * stride + DST]);
        }
        Ok(())
    }

    fn scale_luma(&mut self, frame: &Frame, out: &mut [u8]) {
        let plan = self.plan.take().expect("planned");
        let (data, stride) = (frame.data(0), frame.stride(0));
        self.line_rows = [-1, -1];
        let size = plan.v.size;
        // swscale dithers only sources deeper than 8 bits; others get 64.
        let deep = plan.layout.depth > 8;
        for y in 0..DST {
            let dither: &[u8; 8] = if deep { &DITHER_8X8_128[y & 7] } else { &[64; 8] };
            let first = plan.v.pos[y] as usize;
            let dst = &mut out[y * DST..(y + 1) * DST];
            let a = self.line(&plan, data, stride, first);
            if size == 1 {
                // yuv2plane1_8_c.
                for (x, (o, &v)) in dst.iter_mut().zip(&self.lines[a]).enumerate() {
                    *o = ((v as i32 + dither[x & 7] as i32) >> 7).clamp(0, 255) as u8;
                }
                continue;
            }
            let b = self.line(&plan, data, stride, first + 1);
            let (ca, cb) = (plan.v.coef[y * 2] as i32, plan.v.coef[y * 2 + 1] as i32);
            // yuv2planeX_8_c.
            for x in 0..DST {
                let val = ((dither[x & 7] as i32) << 12)
                    .wrapping_add((self.lines[a][x] as i32).wrapping_mul(ca))
                    .wrapping_add((self.lines[b][x] as i32).wrapping_mul(cb));
                dst[x] = (val >> 19).clamp(0, 255) as u8;
            }
        }
        self.plan = Some(plan);
    }

    /// The horizontally scaled source row `row`, computed at most once per run.
    fn line(&mut self, plan: &Plan, data: &[u8], stride: usize, row: usize) -> usize {
        let r = row as i64;
        if self.line_rows[0] == r {
            return 0;
        }
        if self.line_rows[1] == r {
            return 1;
        }
        // Rows only move down, so the older slot is the one not holding r - 1.
        let slot = if self.line_rows[0] < self.line_rows[1] { 0 } else { 1 };
        let w = plan.w;
        let line = &mut self.lines[slot];
        if let Some(f) = &plan.hfilter {
            // hScale16To15_c: 14-bit taps over native little-endian samples.
            let sh = plan.layout.depth - 1;
            let bytes = &data[row * stride..row * stride + 2 * w];
            let at = |x: usize| u16::from_le_bytes([bytes[2 * x], bytes[2 * x + 1]]) as i32;
            for (i, px) in line.iter_mut().enumerate() {
                let p = f.pos[i] as usize;
                let mut val: i32 = 0;
                for j in 0..f.size {
                    val = val.wrapping_add(at(p + j).wrapping_mul(f.coef[i * f.size + j] as i32));
                }
                *px = (val >> sh).min((1 << 15) - 1) as i16;
            }
        } else {
            // ff_hyscale_fast_c.
            let src = &data[row * stride..row * stride + w];
            let mut xpos: u32 = 0;
            for px in line.iter_mut().take(plan.tail_from) {
                let xx = (xpos >> 16) as usize;
                let alpha = ((xpos & 0xFFFF) >> 9) as i32;
                let a = src[xx] as i32;
                let b = src[xx + 1] as i32;
                *px = ((a << 7) + (b - a) * alpha) as i16;
                xpos = xpos.wrapping_add(plan.x_inc);
            }
            let last = (src[w - 1] as i32 * 128) as i16;
            for px in line.iter_mut().skip(plan.tail_from) {
                *px = last;
            }
        }
        if let Some((coeff, offset)) = self.range {
            for px in line.iter_mut() {
                let y = (*px as i32 * coeff + offset) >> 14;
                *px = y.min((1 << 15) - 1) as i16;
            }
        }
        self.line_rows[slot] = r;
        slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_vertical_filter_reads_inside_the_picture_and_sums_to_one() {
        for h in [8, 100, 128, 129, 240, 480, 576, 720, 1080, 1440, 2160, 4320] {
            let v = vertical_filter(h);
            assert!(v.size >= 1 && v.size <= 2, "{h}: size {}", v.size);
            for y in 0..DST {
                let p = v.pos[y];
                assert!(p >= 0 && (p as usize) + v.size <= h.max(v.size), "{h}: row {y} at {p}");
                let sum: i32 = v.coef[y * v.size..(y + 1) * v.size].iter().map(|&c| c as i32).sum();
                assert_eq!(sum, 4096, "{h}: row {y}");
            }
        }
    }

    #[test]
    fn test_the_range_expansion_matches_libswscale_s_constants() {
        // solve_range_convert(16, 235, 0, 255, 15, 7, 14) in libswscale 8.1.
        let (c, o) = range_to_full();
        assert_eq!(c, 19078);
        assert_eq!(((16 << 7) * c + o) >> 14, -1, "a hair under black, clamped later");
        assert_eq!(((235 << 7) * c + o) >> 14, 255 << 7);
    }

    /// Decodes keyframes from every video under `VID_FP_SCALER_CORPUS` and
    /// compares this module against libswscale's C path (cpu flags forced to
    /// 0, which is what an aarch64 build runs). Ignored: it needs a corpus.
    #[test]
    #[ignore]
    fn test_the_port_matches_libswscale_s_c_path_bit_for_bit() {
        let Ok(root) = std::env::var("VID_FP_SCALER_CORPUS") else { return };
        let per_file: usize = std::env::var("VID_FP_SCALER_FRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
        ffmpeg_next::init().unwrap();
        let mut files: Vec<_> = walk(std::path::Path::new(&root));
        files.sort();
        let (mut frames, mut exact, mut by_format) = (0usize, 0usize, std::collections::BTreeMap::new());
        let mut worst: Vec<(usize, String)> = Vec::new();
        for path in &files {
            let Ok(mut ictx) = ffmpeg_next::format::input(path) else { continue };
            let Some(stream) = ictx.streams().best(ffmpeg_next::media::Type::Video) else { continue };
            let idx = stream.index();
            let Ok(ctx) = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters()) else { continue };
            let Ok(mut dec) = ctx.decoder().video() else { continue };
            let mut n = 0;
            let mut frame = Frame::empty();
            for (s, packet) in ictx.packets() {
                if s.index() != idx || !packet.is_key() {
                    continue;
                }
                if dec.send_packet(&packet).is_err() {
                    continue;
                }
                while dec.receive_frame(&mut frame).is_ok() {
                    if !frame.is_key() && n > 0 {
                        continue;
                    }
                    let mut ours = vec![0u8; DST * DST];
                    let mut d = Downscaler::default();
                    d.run(&frame, &mut ours).unwrap();
                    let theirs = swscale_c(&frame);
                    let diff = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
                    frames += 1;
                    let e = by_format.entry(format!("{:?}", frame.format())).or_insert((0usize, 0usize));
                    e.0 += 1;
                    if diff == 0 {
                        exact += 1;
                        e.1 += 1;
                    } else if worst.len() < 20 {
                        worst.push((diff, format!("{} {:?} {}x{}", path.display(), frame.format(), frame.width(), frame.height())));
                    }
                    n += 1;
                }
                if n >= per_file {
                    break;
                }
            }
        }
        eprintln!("{exact} of {frames} frames bit-identical");
        for (f, (n, e)) in &by_format {
            eprintln!("  {f}: {e}/{n}");
        }
        for (d, what) in &worst {
            eprintln!("  {d} px differ: {what}");
        }
        let ported_mismatch = worst.iter().any(|(_, w)| !w.contains(" PAL8 ") && !w.contains(" RGB"));
        assert!(!ported_mismatch, "a frame this module scales itself differs from the C path");
    }

    fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(dir) else { return out };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name();
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            if p.is_dir() {
                out.extend(walk(&p));
            } else {
                out.push(p);
            }
        }
        out
    }

    fn swscale_c(frame: &Frame) -> Vec<u8> {
        unsafe { ffmpeg_next::ffi::av_force_cpu_flags(0) };
        let mut ctx = ffmpeg_next::software::scaling::context::Context::get(
            frame.format(),
            frame.width(),
            frame.height(),
            Pixel::GRAY8,
            DST as u32,
            DST as u32,
            ffmpeg_next::software::scaling::flag::Flags::FAST_BILINEAR,
        )
        .unwrap();
        unsafe { ffmpeg_next::ffi::av_force_cpu_flags(-1) };
        let mut out = Frame::empty();
        ctx.run(frame, &mut out).unwrap();
        let mut v = vec![0u8; DST * DST];
        for y in 0..DST {
            v[y * DST..(y + 1) * DST].copy_from_slice(&out.data(0)[y * out.stride(0)..][..DST]);
        }
        v
    }
}
