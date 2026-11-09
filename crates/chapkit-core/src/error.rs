use chapkit_dataframe::FrameError;

/// Errors a model or the core contract can raise. Carries no HTTP knowledge;
/// the service layer maps these onto problem+json responses or job failures.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The caller supplied unusable input (missing columns, empty frames, bad horizon).
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// The config could not be read as the model's config type.
    #[error("invalid config: {0}")]
    Config(String),
    /// The model itself failed.
    #[error("model error: {0}")]
    Model(String),
    /// Table shape errors.
    #[error(transparent)]
    Frame(#[from] FrameError),
    /// (De)serialization of configs or trained models failed.
    #[error("serialization: {0}")]
    Serde(String),
}

impl Error {
    /// Shorthand for [`Error::InvalidInput`].
    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::InvalidInput(msg.into())
    }

    /// Shorthand for [`Error::Model`].
    pub fn model(msg: impl Into<String>) -> Self {
        Error::Model(msg.into())
    }
}

/// Result alias for the core crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;
