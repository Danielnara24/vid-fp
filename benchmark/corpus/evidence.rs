//! Benchmark helper (not shipped): show the frames behind a link.
//!
//!     evidence OUT_DIR CACHE D P A B
//!
//! Runs the scan the window runs on just A and B, then writes the matched
//! keyframe pairs it reports as OUT_DIR/NN_a.ppm / NN_b.ppm and a pairs.tsv
//! of (t_a_ms, t_b_ms, distance). Times are each fingerprint's own clock.
use std::io::Write;

fn ppm(path: &std::path::Path, (w, h, rgba): (u32, u32, Vec<u8>)) -> anyhow::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "P6\n{w} {h}\n255\n")?;
    for px in rgba.chunks_exact(4) {
        f.write_all(&px[..3])?;
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (out, cache, d, p, fa, fb) = (&a[1], &a[2], &a[3], &a[4], &a[5], &a[6]);
    std::fs::create_dir_all(out)?;
    vid_fp::thumbnail::init()?;
    let result = std::path::Path::new(out).join("found.json");
    let argv = ["vid-fp", fa, fb, "-d", d, "-p", p, "--cache", cache, "-q"];
    vid_fp::worker_main(&result, argv.iter().map(std::ffi::OsString::from))?;
    let found = vid_fp::found::Found::load(&result)?;
    let mut tsv = std::fs::File::create(std::path::Path::new(out).join("pairs.tsv"))?;
    let Some(group) = found.groups.first() else {
        writeln!(tsv, "# no link")?;
        return Ok(());
    };
    // The comparison held by the member that is not the representative.
    for m in &group.members {
        for c in &m.compared {
            let this = &found.files[m.file].path;
            let other = &found.files[c.with].path;
            // Write it from A's side whichever way round it was stored.
            let flip = this != fa;
            writeln!(tsv, "# matched_frames {}  tolerance {}", c.matched_frames, c.tolerance)?;
            for (k, &(ta, tb, dist)) in c.pairs.iter().enumerate() {
                let (t_a, t_b) = if flip { (tb, ta) } else { (ta, tb) };
                let (pa, pb) = if flip { (other, this) } else { (this, other) };
                writeln!(tsv, "{t_a}\t{t_b}\t{dist}")?;
                ppm(&std::path::Path::new(out).join(format!("{k:02}_a.ppm")), vid_fp::thumbnail::frame_at(pa, t_a, 320, 240)?)?;
                ppm(&std::path::Path::new(out).join(format!("{k:02}_b.ppm")), vid_fp::thumbnail::frame_at(pb, t_b, 320, 240)?)?;
            }
            return Ok(());
        }
    }
    Ok(())
}
