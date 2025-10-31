use std::fs;
use std::io;
use std::os::unix::fs::MetadataExt;
// use std::os::macos::fs::MetadataExt;

fn main() -> io::Result<()> {
    let path = "testing.txt"; // Replace with your file path

    let metadata = fs::metadata(path)?;

    // Accessing metadata fields
    println!("File size: {} bytes", metadata.len());
    println!("Is directory: {}", metadata.is_dir());
    println!("Is file: {}", metadata.is_file());
    // println!("mTime: {}", metadata.mtime());
    if let Ok(time) = metadata.created() {
        println!("inode: {:?}", metadata.ino());
        println!("Device: {:?}", metadata.dev());
        println!("File type: {:?}", metadata.file_type());
        println!("Permissions: {:?}", metadata.permissions());
        println!("Created actual: {:?}", time);
        println!("Created: {:?}", chrono::DateTime::<chrono::Local>::from(time).timestamp());
    } else {
        println!("Not supported on this platform or filesystem");
    }

    println!("Last accessed: {:?}", metadata.modified as i64);

    if let Ok(modified_time) = metadata.modified() {
        println!("Last modified: {:?}", modified_time);
    } else {
        println!("Last modified time not available on this platform.");
    }

    Ok(())
}