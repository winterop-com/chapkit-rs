//! Record ids: ULIDs that always increase within the process, so a listing ordered by id is in
//! creation order, even for records made in the same millisecond (plain ULIDs from one
//! millisecond differ only in random bits, so their order would be arbitrary).

use std::sync::{Mutex, PoisonError};

use ulid::{Generator, Ulid};

static GENERATOR: Mutex<Generator> = Mutex::new(Generator::new());

/// A new id, larger than every id this process made before it.
pub fn new_id() -> Ulid {
    let mut generator = GENERATOR.lock().unwrap_or_else(PoisonError::into_inner);
    match generator.generate() {
        Ok(id) => id,
        // 2^80 ids in one millisecond: carry on with a fresh random part.
        Err(overflow) => overflow.commit_overflow_random(),
    }
}

#[cfg(test)]
mod tests {
    use super::new_id;

    #[test]
    fn ids_made_in_a_burst_are_in_order() {
        let ids: Vec<_> = (0..10_000).map(|_| new_id()).collect();
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
    }
}
