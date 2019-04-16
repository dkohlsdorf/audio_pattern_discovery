use std::io::*;

#[derive(Debug)]
pub enum DiscoveryError {
    IO(Error),
}

impl From<Error> for DiscoveryError {
    fn from(e: Error) -> DiscoveryError {
        DiscoveryError::IO(e)
    }
}

pub type Result<T> = std::result::Result<T, DiscoveryError>;
