use std::io::{self, BufRead, BufReader, Write};
use std::{env, fs::File, process};

fn main() {
    let args = parse_args();
    let program_name = &args[0];
    match args[1].as_str() {
        "-h" | "--help" => print_help(program_name),
        "set" => {
            let (Some(key), Some(value)) = (&args.get(2), args.get(3)) else {
                eprintln!("key and value must not be empty");
                return;
            };
            _ = handle_set(key.as_str(), value.as_str());
        }
        "get" => {
            let Some(key) = args.get(2) else {
                eprintln!("key must be passed");
                return;
            };

            let result = handle_get(key);
            match result {
                Ok(Some(val)) => println!("{val}"),
                Ok(None) => println!("key not found"),
                Err(e) => eprintln!("error fetchig data for given key. {e}"),
            }
        }
        _ => {
            print_help(program_name);
        }
    }
}

/* fn create_file() {
    println!("creating data.log file");
    File::create_new("data.log");
} */

fn parse_args() -> Vec<String> {
    let args: Vec<_> = env::args().collect();
    println!("{:?}", args);
    let program_name = &args[0];

    if args.len() < 2 {
        eprintln!("Missing required arguments");
        print_help(program_name);
        process::exit(1);
    }

    args
}

fn print_help(program_name: &String) {
    println!(
        "a simple key value store, \n use {} set <key> <value> to store \n use {} get <key> to retrieve data",
        program_name, program_name
    );
}

fn handle_set(key: &str, value: &str) -> io::Result<()> {
    let mut file = File::options().append(true).create(true).open("data.log")?;

    println!("appending key: {}, value: {}", key, value);
    writeln!(file, "{}, {}", key, value)?;
    Ok(())
}

fn handle_get(key: &str) -> io::Result<Option<String>> {
    let file = File::open("data.log")?;
    let reader = BufReader::new(file);
    let mut last_match = None;

    for line in reader.lines() {
        let line = line?;
        // if line.starts_with(key) {
        //     last_match = Some(line);
        // }

        if let Some((k, v)) = line.split_once(' ') {
            if k.trim() == key.trim() {
                last_match = Some(v.trim().to_string());
            }
        }
    }
    Ok(last_match)
}
