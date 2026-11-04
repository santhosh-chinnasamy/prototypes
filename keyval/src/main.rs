use std::{env, fs::File, process};

fn main() {
    create_file();
    parse_args();
}

fn create_file() {
    println!("creating data.log file");
    File::create_new("data.log");
}

fn parse_args() {
    let args: Vec<_> = env::args().collect();
    let program_name = &args[0];

    if *&args.len() < 2 {
        eprintln!("Missing required arguments");
        print_help(program_name);
        process::exit(1);
    }

    match args[1].as_str() {
        "-h" | "--help" => print_help(program_name),
        "set" => {
            println!("appending");
        }
        "get" => {
            println!("fetching");
        }
        _ => {
            print_help(program_name);
        }
    }
}

fn print_help(program_name: &String) {
    println!(
        "a simple key value store, \n use {} set <key> <value> to store \n use {} get <key> to retrieve data",
        &program_name, &program_name
    );
}
