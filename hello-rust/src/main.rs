use ferris_says::say;
use std::env::current_dir;
use std::fs::File;
use std::io::stdout;
use std::io::BufWriter;
use std::io::Read;
use std::str::from_utf8;

fn hello() {
    println!("Hello, world!");
    let stdout = stdout();
    let message = String::from("Hello from Santhosh");
    let width = message.chars().count();
    print!("{}", &message.chars().count());

    let mut writer = BufWriter::new(stdout.lock());
    say(&message, width, &mut writer).unwrap();
}

fn read_file() {
    let current_dir = current_dir().unwrap().display().to_string() + "/src";
    let path = current_dir + "/hello.txt";
    let mut file = match File::open(path) {
        Err(error) => panic!("Error opening the file: {}", error),
        Ok(value) => value,
    };

    let stat = match file.metadata() {
        Err(error) => panic!("Error getting metadata: {}", error),
        Ok(value) => value,
    };

    let mut buffer = vec![0; stat.len() as usize];

    match file.read(&mut buffer) {
        Err(error) => panic!("Error reading the file: {}", error),
        Ok(_) => (),
    };

    let data = match from_utf8(&buffer) {
        Err(error) => panic!("Error converting to string: {}", error),
        Ok(value) => value,
    };

    println!("Content is: {}", data);
}

fn main() {
    hello();
    read_file();
}
