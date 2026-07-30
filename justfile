test e2e:
    cargo test --test conformance -- --ignored

flush-db:
    cargo run -- flush-db