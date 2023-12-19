all:
	cargo b --all-targets
	
test:
	cargo test 

lint:
	cargo clippy --all-targets

fix:
	cargo clippy --fix --allow-dirty

fmt:
	cargo +nightly fmt
