# Developer Onboarding

## Prerequisites
- [Rust](https://www.rust-lang.org/) 1.75 or newer
- `cargo fmt` and `clippy` installed via `rustup component add rustfmt clippy`
- Backend dependencies as needed (see [backend-setup](backend-setup.md))

## Getting Started
1. Fork and clone the repository.
2. Install required backend dependencies.
3. Run the test suite:
   ```sh
   cargo test --all --exclude amduda
   ```
4. Format code with `cargo fmt` and lint with `cargo clippy` before submitting patches.

## Contribution Guidelines
- Use descriptive commit messages and keep patches focused.
- Document new APIs and update relevant docs.
- Run the tests and linters before opening a pull request.

Happy hacking!
