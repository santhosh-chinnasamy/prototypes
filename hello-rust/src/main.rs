use ferris_says::say;
use std::env::current_dir;
use std::fs::read_to_string;
use std::io::stdout;
use std::io::BufWriter;

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
    match read_to_string(path) {
        Err(err) => panic!("Error reading file:{}", err),
        Ok(contents) => println!("content: {}", contents),
    }
}

fn main() {
    hello();
    read_file();
}
