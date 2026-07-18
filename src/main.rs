mod compare;
mod fingerprint;

use compare::compare_videos;
use fingerprint::fingerprint_video;

fn main() {
    let vid1_path = "/home/daniel/Documents/AN/2-3.mp4";
    let vid2_path = "/home/daniel/Documents/AN/3.mp4";

    println!("--- Fingerprinting Video 1 ---");
    let fp1_opt = fingerprint_video(vid1_path);
    
    println!("\n--- Fingerprinting Video 2 ---");
    let fp2_opt = fingerprint_video(vid2_path);

    if let (Some(fp1), Some(fp2)) = (fp1_opt, fp2_opt) {
        println!("\n--- Comparing Videos ---");
        // Using your default max_hamming=5 and min_match=0.15
        if let Some(result) = compare_videos(&fp1, &fp2, 5, 0.15) {
            println!("\n>>> MATCH FOUND! <<<");
            println!("Match Length: {:.2} seconds", result.match_length_seconds);
            println!("Match Percent: {:.2}%", result.match_percent * 100.0);
            println!("Interval A: {:.2}s to {:.2}s", result.interval_a.0, result.interval_a.1);
            println!("Interval B: {:.2}s to {:.2}s", result.interval_b.0, result.interval_b.1);
        } else {
            println!("\n>>> NO MATCH <<< (Below threshold or no overlapping frames)");
        }
    } else {
        eprintln!("Error: Could not extract fingerprints from one or both videos.");
    }
}