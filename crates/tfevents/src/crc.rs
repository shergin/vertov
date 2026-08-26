//! CRC-32C (Castagnoli) with TFRecord's LevelDB-style masking.
//!
//! TFRecord checksums are CRC-32C values run through a masking permutation
//! (inherited from LevelDB) so that a CRC stored alongside the data it covers
//! does not itself look like coverable data:
//! `mask(c) = ((c >> 15) | (c << 17)) + 0xa282ead8` (wrapping).

/// A masked CRC-32C checksum as stored on disk in TFRecord framing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MaskedCrc(pub u32);

impl MaskedCrc {
    /// Computes the masked CRC-32C of `data`.
    pub fn compute(data: &[u8]) -> MaskedCrc {
        MaskedCrc(mask(crc32c(data)))
    }
}

// Reflected polynomial for CRC-32C (Castagnoli, 0x1EDC6F41).
const POLY: u32 = 0x82F63B78;

const TABLE: [u32; 256] = build_table();

const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ POLY } else { crc >> 1 };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Computes the (unmasked) CRC-32C of `data`.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc = (crc >> 8) ^ TABLE[((crc ^ u32::from(byte)) & 0xFF) as usize];
    }
    !crc
}

const MASK_DELTA: u32 = 0xa282_ead8;

fn mask(crc: u32) -> u32 {
    crc.rotate_right(15).wrapping_add(MASK_DELTA)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32c_known_vectors() {
        // RFC 3720 test vectors.
        assert_eq!(crc32c(&[0u8; 32]), 0x8A91_36AA);
        assert_eq!(crc32c(&[0xFFu8; 32]), 0x62A8_AB43);
        assert_eq!(crc32c(b""), 0);
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn masked_crc_matches_rustboard_vector() {
        // Test vector from TensorBoard's Rust data server.
        assert_eq!(
            MaskedCrc::compute(b"\x1a\x11CRC test, one two"),
            MaskedCrc(0x5794_d08a)
        );
    }

    #[test]
    fn mask_is_injective_on_samples() {
        let inputs: &[&[u8]] = &[b"", b"a", b"ab", b"abc", b"\x00", b"\x00\x00"];
        let mut seen = std::collections::HashSet::new();
        for input in inputs {
            assert!(seen.insert(MaskedCrc::compute(input).0));
        }
    }
}
