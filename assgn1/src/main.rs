use std::env;
use std::path::PathBuf;

use ffmpeg_sidecar::command::FfmpegCommand;
use workspace_root::get_workspace_root;

use std::fs::File;
use std::io::BufReader;
use std::io::{BufWriter, Write};

use bitbit::BitReader;
use bitbit::BitWriter;
use bitbit::MSB;

use toy_ac::decoder::Decoder;
use toy_ac::encoder::Encoder;
use toy_ac::symbol_model::VectorCountSymbolModel;

use ffmpeg_sidecar::event::StreamTypeSpecificData::Video;

const NUM_CONTEXTS: usize = 64;

/// Temporal MED predictor.
/// Applies the Median Edge Detector to temporal differences of causal neighbors
/// (left, top, top-left). MED is edge-adaptive: at horizontal edges it uses the
/// top temporal diff, at vertical edges it uses the left temporal diff, and in
/// smooth regions it uses the gradient (left + top - top_left).
fn temporal_med_predict(decoded_current: &[u8], prior_frame: &[u8], width: u32, r: u32, c: u32) -> u8 {
    let prior_val = prior_frame[(r * width + c) as usize] as i32;

    let left_tdiff = if c > 0 {
        let idx = (r * width + c - 1) as usize;
        decoded_current[idx] as i32 - prior_frame[idx] as i32
    } else {
        0
    };

    let top_tdiff = if r > 0 {
        let idx = ((r - 1) * width + c) as usize;
        decoded_current[idx] as i32 - prior_frame[idx] as i32
    } else {
        0
    };

    let tl_tdiff = if r > 0 && c > 0 {
        let idx = ((r - 1) * width + c - 1) as usize;
        decoded_current[idx] as i32 - prior_frame[idx] as i32
    } else {
        0
    };

    // MED: median of (left_tdiff, top_tdiff, left_tdiff + top_tdiff - tl_tdiff)
    let combined = left_tdiff + top_tdiff - tl_tdiff;
    let adjustment = if tl_tdiff >= left_tdiff.max(top_tdiff) {
        left_tdiff.min(top_tdiff)
    } else if tl_tdiff <= left_tdiff.min(top_tdiff) {
        left_tdiff.max(top_tdiff)
    } else {
        combined
    };

    (prior_val + adjustment).clamp(0, 255) as u8
}

/// Compute local temporal activity from left and top neighbors.
/// Returns (left_activity, top_activity) as quantized bin indices.
/// Used together for 2D context selection: left_bin * 8 + top_bin = 0..63.
fn activity_bins(decoded_current: &[u8], prior_frame: &[u8], width: u32, r: u32, c: u32) -> usize {
    let left_abs = if c > 0 {
        let idx = (r * width + c - 1) as usize;
        (decoded_current[idx] as i32 - prior_frame[idx] as i32).unsigned_abs() as u8
    } else {
        0
    };

    let top_abs = if r > 0 {
        let idx = ((r - 1) * width + c) as usize;
        (decoded_current[idx] as i32 - prior_frame[idx] as i32).unsigned_abs() as u8
    } else {
        0
    };

    let left_bin = quantize8(left_abs);
    let top_bin = quantize8(top_abs);
    left_bin * 8 + top_bin
}

/// Quantize an absolute difference (0..255) into one of 8 bins.
fn quantize8(val: u8) -> usize {
    match val {
        0..=1 => 0,
        2..=3 => 1,
        4..=7 => 2,
        8..=15 => 3,
        16..=31 => 4,
        32..=63 => 5,
        64..=127 => 6,
        _ => 7,
    }
}

/// Convert a mod-256 residual to signed [-128, 127].
fn to_signed(residual: i32) -> i32 {
    if residual > 127 { residual - 256 } else { residual }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Make sure ffmpeg is installed
    ffmpeg_sidecar::download::auto_download().unwrap();

    // Command line options
    // -verbose, -no_verbose                Default: -no_verbose
    // -report, -no_report                  Default: -report
    // -check_decode, -no_check_decode      Default: -no_check_decode
    // -skip_count n                        Default: -skip_count 0
    // -count n                             Default: -count 10
    // -in file_path                        Default: bourne.mp4 in data subdirectory of workplace
    // -out file_path                       Default: out.dat in data subdirectory of workplace

    // Set up default values of options
    let mut verbose = false;
    let mut report = true;
    let mut check_decode = false;
    let mut skip_count = 0;
    let mut count = 10;

    let mut data_folder_path = get_workspace_root();
    data_folder_path.push("data");

    let mut input_file_path = data_folder_path.join("bourne.mp4");
    let mut output_file_path = data_folder_path.join("out.dat");

    parse_args(
        &mut verbose,
        &mut report,
        &mut check_decode,
        &mut skip_count,
        &mut count,
        &mut input_file_path,
        &mut output_file_path,
    );

    // Run an FFmpeg command to decode video from inptu_file_path
    // Get output as grayscale (i.e., just the Y plane)

    let mut iter = FfmpegCommand::new() // <- Builder API like `std::process::Command`
        .input(input_file_path.to_str().unwrap())
        .format("rawvideo")
        .pix_fmt("gray8")
        .output("-")
        .spawn()? // <- Ordinary `std::process::Child`
        .iter()?; // <- Blocking iterator over logs and output

    // Figure out geometry of frame.
    let mut width = 0;
    let mut height = 0;

    let metadata = iter.collect_metadata()?;
    for i in 0..metadata.output_streams.len() {
        match &metadata.output_streams[i].type_specific_data {
            Video(vid_stream) => {
                width = vid_stream.width;
                height = vid_stream.height;

                if verbose {
                    println!(
                        "Found video stream at output stream index {} with dimensions {} x {}",
                        i, width, height
                    );
                }
                break;
            }
            _ => (),
        }
    }
    assert!(width != 0);
    assert!(height != 0);

    // Set up initial prior frame as uniform medium gray (y = 128)
    let mut prior_frame = vec![128 as u8; (width * height) as usize];

    let output_file = match File::create(&output_file_path) {
        Err(_) => panic!("Error opening output file"),
        Ok(f) => f,
    };

    // Setup bit writer and arithmetic encoder.

    let mut buf_writer = BufWriter::new(output_file);
    let mut bw = BitWriter::new(&mut buf_writer);

    let mut enc = Encoder::new();

    // Set up arithmetic coding contexts (64 contexts: 8 left bins x 8 top bins)
    let mut contexts: Vec<VectorCountSymbolModel<i32>> = (0..NUM_CONTEXTS)
        .map(|_| VectorCountSymbolModel::new((0..=255).collect()))
        .collect();

    // Bias correction: track average signed prediction error per context
    let mut bias_sum = [0i32; NUM_CONTEXTS];
    let mut bias_count = [0i32; NUM_CONTEXTS];

    // Process frames
    for frame in iter.filter_frames() {
        if frame.frame_num < skip_count {
            if verbose {
                println!("Skipping frame {}", frame.frame_num);
            }
        } else if frame.frame_num < skip_count + count {
            let current_frame: Vec<u8> = frame.data; // <- raw pixel y values

            let bits_written_at_start = enc.bits_written();

            // Process pixels in row major order.
            for r in 0..height {
                for c in 0..width {
                    let pixel_index = (r * width + c) as usize;

                    // Gradient-adjusted temporal predictor
                    let base_pred = temporal_med_predict(&current_frame, &prior_frame, width, r, c) as i32;

                    // Select context based on 2D local activity (left x top)
                    let ctx = activity_bins(&current_frame, &prior_frame, width, r, c);

                    // Apply bias correction to prediction
                    let correction = if bias_count[ctx] > 0 { bias_sum[ctx] / bias_count[ctx] } else { 0 };
                    let adjusted_pred = (base_pred + correction).clamp(0, 255);

                    // Compute residual (mod 256 for lossless)
                    let pixel_difference = ((current_frame[pixel_index] as i32 - adjusted_pred) + 256) % 256;

                    enc.encode(&pixel_difference, &contexts[ctx], &mut bw);
                    contexts[ctx].incr_count(&pixel_difference);

                    // Update bias tracking
                    let signed_res = to_signed(pixel_difference);
                    bias_sum[ctx] += signed_res;
                    bias_count[ctx] += 1;
                    if bias_count[ctx] >= 256 {
                        bias_sum[ctx] /= 2;
                        bias_count[ctx] /= 2;
                    }
                }
            }

            prior_frame = current_frame;

            let bits_written_at_end = enc.bits_written();

            if verbose {
                println!(
                    "frame: {}, compressed size (bits): {}",
                    frame.frame_num,
                    bits_written_at_end - bits_written_at_start
                );
            }
        } else {
            break;
        }
    }

    // Tie off arithmetic encoder and flush to file.
    enc.finish(&mut bw)?;
    bw.pad_to_byte()?;
    buf_writer.flush()?;

    // Decompress and check for correctness.
    if check_decode {
        let output_file = match File::open(&output_file_path) {
            Err(_) => panic!("Error opening output file"),
            Ok(f) => f,
        };
        let mut buf_reader = BufReader::new(output_file);
        let mut br: BitReader<_, MSB> = BitReader::new(&mut buf_reader);

        let iter = FfmpegCommand::new() // <- Builder API like `std::process::Command`
            .input(input_file_path.to_str().unwrap())
            .format("rawvideo")
            .pix_fmt("gray8")
            .output("-")
            .spawn()? // <- Ordinary `std::process::Child`
            .iter()?; // <- Blocking iterator over logs and output

        let mut dec = Decoder::new();

        let mut contexts: Vec<VectorCountSymbolModel<i32>> = (0..NUM_CONTEXTS)
            .map(|_| VectorCountSymbolModel::new((0..=255).collect()))
            .collect();

        let mut bias_sum = [0i32; NUM_CONTEXTS];
        let mut bias_count = [0i32; NUM_CONTEXTS];

        // Set up initial prior frame as uniform medium gray
        let mut prior_frame = vec![128 as u8; (width * height) as usize];

        'outer_loop: 
        for frame in iter.filter_frames() {
            if frame.frame_num < skip_count + count {
                if verbose {
                    print!("Checking frame: {} ... ", frame.frame_num);
                }

                let current_frame: Vec<u8> = frame.data; // <- raw pixel y values

                // Reconstruct decoded frame pixel by pixel
                let mut decoded_frame = vec![0u8; (width * height) as usize];

                // Process pixels in row major order.
                for r in 0..height {
                    for c in 0..width {
                        let pixel_index = (r * width + c) as usize;

                        // Same gradient-adjusted predictor (using decoded_frame for causal neighbors)
                        let base_pred = temporal_med_predict(&decoded_frame, &prior_frame, width, r, c) as i32;

                        // Same 2D context selection
                        let ctx = activity_bins(&decoded_frame, &prior_frame, width, r, c);

                        // Same bias correction
                        let correction = if bias_count[ctx] > 0 { bias_sum[ctx] / bias_count[ctx] } else { 0 };
                        let adjusted_pred = (base_pred + correction).clamp(0, 255);

                        let decoded_pixel_difference = dec.decode(&contexts[ctx], &mut br).to_owned();
                        contexts[ctx].incr_count(&decoded_pixel_difference);

                        let pixel_value = (adjusted_pred + decoded_pixel_difference + 256) % 256;
                        decoded_frame[pixel_index] = pixel_value as u8;

                        // Update bias (same as encoder)
                        let signed_res = to_signed(decoded_pixel_difference);
                        bias_sum[ctx] += signed_res;
                        bias_count[ctx] += 1;
                        if bias_count[ctx] >= 256 {
                            bias_sum[ctx] /= 2;
                            bias_count[ctx] /= 2;
                        }

                        if pixel_value != current_frame[pixel_index] as i32 {
                            println!(
                                " error at ({}, {}), should decode {}, got {}",
                                c, r, current_frame[pixel_index], pixel_value
                            );
                            println!("Abandoning check of remaining frames");
                            break 'outer_loop;
                        }
                    }
                }
                println!("correct.");
                prior_frame = current_frame;
            } else {
                break 'outer_loop;
            }
        }
    }

    // Emit report
    if report {
        println!(
            "{} frames encoded, average size (bits): {}, compression ratio: {:.2}",
            count,
            enc.bits_written() / count as u64,
            (width * height * 8 * count) as f64 / enc.bits_written() as f64
        )
    }

    Ok(())
}

fn parse_args(
    verbose: &mut bool,
    report: &mut bool,
    check_decode: &mut bool,
    skip_count: &mut u32,
    count: &mut u32,
    input_file_path: &mut PathBuf,
    output_file_path: &mut PathBuf,
) -> () {
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg == "-verbose" {
            *verbose = true;
        } else if arg == "-no_verbose" {
            *verbose = false;
        } else if arg == "-report" {
            *report = true;
        } else if arg == "-no_report" {
            *report = false;
        } else if arg == "-check_decode" {
            *check_decode = true;
        } else if arg == "-no_check_decode" {
            *check_decode = false;
        } else if arg == "-skip_count" {
            match args.next() {
                Some(skip_count_string) => {
                    *skip_count = skip_count_string.parse::<u32>().unwrap();
                }
                None => {
                    panic!("Expected count after -skip_count option");
                }
            }
        } else if arg == "-count" {
            match args.next() {
                Some(count_string) => {
                    *count = count_string.parse::<u32>().unwrap();
                }
                None => {
                    panic!("Expected count after -count option");
                }
            }
        } else if arg == "-in" {
            match args.next() {
                Some(input_file_path_string) => {
                    *input_file_path = PathBuf::from(input_file_path_string);
                }
                None => {
                    panic!("Expected input file name after -in option");
                }
            }
        } else if arg == "-out" {
            match args.next() {
                Some(output_file_path_string) => {
                    *output_file_path = PathBuf::from(output_file_path_string);
                }
                None => {
                    panic!("Expected output file name after -out option");
                }
            }
        }
    }
}
