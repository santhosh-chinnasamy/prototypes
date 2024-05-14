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

fn read_file(path: &str) -> String {
    let current_dir = current_dir().unwrap().display().to_string() + "/src";
    let file_path = current_dir + path;
    let data = match read_to_string(file_path) {
        Err(err) => {
            let message = format!("Err reading file @ {} and error: \n {}", path, err);
            panic!("{}", message)
        },
        Ok(contents) => contents,
    };

    return data;
}

fn read_files() {

    let hello = read_file("/hello.txt");
    let world = read_file("/worlds.txt");
    println!("{} ---- {}", hello, world);
}



fn main() {
    hello();
    read_files();
}
