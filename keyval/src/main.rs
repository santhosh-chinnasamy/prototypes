use std::io::{self, BufRead, BufReader, Seek, Write};

fn main() {
    println!("keyval");
    println!("run help for more info");
    let mut file = File::options()
        .read(true)
        .append(true)
        .create(true)
        .open("data.log")
        .expect("data.log file Error");

    loop {
        print!(">");
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        let bytes_read = io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");

        if bytes_read == 0 {
            println!("\n Bye");
            break;
        }

        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        let mut parsed = parse_input(input);

        let command = parsed.next();

        match command {
            Some("set") => {
                let key = parsed.next();
                let value = parsed.next();
                match (key, value) {
                    (Some(key), Some(value)) => {
                        handle_set(&mut file, key, value);
                    }
                    _ => eprintln!("Usage: set <key> <value>"),
                }
            }
            Some("get") => {
                let key = parsed.next();
                match key {
                    Some(key) => {
                        let result = handle_get(&mut file, key);
                        match result {
                            Ok(Some(value)) => println!("{}", value),
                            Ok(None) => println!("Key not found"),
                            Err(err) => eprintln!("Error reading file: {}", err),
                        }
                    }
                    None => eprintln!("Usage: get <key>"),
                }
            }
            Some("quit" | "exit") => {
                println!("Bye!");
                break;
            }
            _ => {
                print_help();
            }
        }
    }
}

fn parse_input(input: &str) -> std::str::SplitWhitespace<'_> {
    input.split_whitespace()
}

fn print_help() {
    println!(
        "> a simple key value store, \n
        use set <key> <value> to store \n
        use get <key> to retrieve data"
    );
}

fn handle_set(file: &mut File, key: &str, value: &str) -> io::Result<()> {
    if key.is_empty() || value.is_empty() {
        eprint!("missing key or value");
        return Ok(());
    };

    println!("appending key: {}, value: {}", key, value);
    writeln!(file, "{} {}", key, value)?;
    Ok(())
}

fn handle_get(file: &mut File, key: &str) -> io::Result<Option<String>> {
    file.seek(io::SeekFrom::Start(0))?;
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
