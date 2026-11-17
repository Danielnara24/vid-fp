//! `vid-fp`: the command line. Everything it does is the library's (`lib.rs`),
//! which the desktop app (`src/gui`) runs too.

fn main() -> anyhow::Result<()> {
    vid_fp::main()
}
