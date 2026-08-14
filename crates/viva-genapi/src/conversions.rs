//! Numeric conversion utilities for register values and bitfields.

use viva_genapi_xml::{ByteOrder, Sign};

use crate::GenApiError;
use crate::bitops::BitOpsError;
use crate::nodes::FloatNode;

/// Convert a byte slice (up to 8 bytes) to a 64-bit integer, per `order`.
///
/// `sign` decides whether the payload's top bit means "negative" or is just
/// another value bit. GenICam defaults to [`Sign::Unsigned`], and getting it
/// wrong is silent right up until a register's top bit is set: a
/// `GevCurrentIPAddress` of `192.168.1.160` (`0xC0A801A0`) then reads as
/// `-1062731360`.
///
/// `order` must be the node's own declared `<Endianess>` (GenICam integer
/// registers default to `BigEndian` when the tag is absent, same as
/// `<Float>`/`<FloatReg>` — see `decode_ieee754`). Hardcoding big-endian here
/// regardless of `order` was a real bug: it silently misdecoded every
/// `LittleEndian`-declared `<Integer>`/`<IntReg>` node, confirmed against a
/// real Teledyne DALSA Genie Nano whose Width/Height/OffsetX/OffsetY/
/// WidthMax/HeightMax registers are all declared `LittleEndian` and all
/// decoded wrong before this fix.
pub fn bytes_to_i64(name: &str, bytes: &[u8], sign: Sign, order: ByteOrder) -> Result<i64, GenApiError> {
    if bytes.is_empty() {
        return Err(GenApiError::Parse(format!(
            "node {name} returned empty payload"
        )));
    }
    if bytes.len() > 8 {
        return Err(GenApiError::Parse(format!(
            "node {name} uses unsupported width {}",
            bytes.len()
        )));
    }
    let mut buf = [0u8; 8];
    let value = match order {
        ByteOrder::Big => {
            let offset = 8 - bytes.len();
            buf[offset..].copy_from_slice(bytes);
            if sign.is_signed() && (bytes[0] & 0x80) != 0 {
                for byte in &mut buf[..offset] {
                    *byte = 0xFF;
                }
            }
            i64::from_be_bytes(buf)
        }
        ByteOrder::Little => {
            buf[..bytes.len()].copy_from_slice(bytes);
            // The sign bit lives in the *last* byte of a little-endian
            // payload (its most significant byte), not the first.
            if sign.is_signed() && (bytes[bytes.len() - 1] & 0x80) != 0 {
                for byte in &mut buf[bytes.len()..] {
                    *byte = 0xFF;
                }
            }
            i64::from_le_bytes(buf)
        }
    };
    // A full-width unsigned register can hold values an i64 cannot. GenApi's
    // IInteger is int64, so there is nowhere to put them; say so rather than
    // hand back a negative number.
    if !sign.is_signed() && bytes.len() == 8 && value < 0 {
        return Err(GenApiError::Parse(format!(
            "node {name} holds an unsigned 64-bit value larger than i64::MAX"
        )));
    }
    Ok(value)
}

/// Convert a 64-bit integer to a byte vector of the given width, per `order`.
pub fn i64_to_bytes(
    name: &str,
    value: i64,
    width: u32,
    sign: Sign,
    order: ByteOrder,
) -> Result<Vec<u8>, GenApiError> {
    if width == 0 || width > 8 {
        return Err(GenApiError::Parse(format!(
            "node {name} has unsupported width {width}"
        )));
    }
    let width_usize = width as usize;
    let data = match order {
        ByteOrder::Big => value.to_be_bytes()[8 - width_usize..].to_vec(),
        ByteOrder::Little => value.to_le_bytes()[..width_usize].to_vec(),
    };
    let roundtrip = bytes_to_i64(name, &data, sign, order)?;
    if roundtrip != value {
        return Err(GenApiError::Range(format!(
            "value {value} does not fit {width} bytes for {name}"
        )));
    }
    Ok(data)
}

/// Interpret an extracted bitfield value, applying sign extension if needed.
pub fn interpret_bitfield_value(
    name: &str,
    raw: u64,
    bit_length: u16,
    signed: bool,
) -> Result<i64, GenApiError> {
    if signed {
        Ok(sign_extend(raw, bit_length))
    } else {
        i64::try_from(raw).map_err(|_| {
            GenApiError::Parse(format!(
                "bitfield value {raw} exceeds i64 range for node {name}"
            ))
        })
    }
}

/// Encode a value into bitfield representation, validating range constraints.
pub fn encode_bitfield_value(
    name: &str,
    value: i64,
    bit_length: u16,
    signed: bool,
) -> Result<u64, GenApiError> {
    if bit_length == 0 || bit_length > 64 {
        return Err(GenApiError::Parse(format!(
            "node {name} uses unsupported bitfield width {bit_length}"
        )));
    }
    if signed {
        let width = bit_length as u32;
        let min_allowed = -(1i128 << (width - 1));
        let max_allowed = (1i128 << (width - 1)) - 1;
        let value_i128 = value as i128;
        if value_i128 < min_allowed || value_i128 > max_allowed {
            return Err(GenApiError::ValueTooWide {
                name: name.to_string(),
                value,
                bit_length,
            });
        }
        let mask = mask_u128(bit_length) as i128;
        Ok((value_i128 & mask) as u64)
    } else {
        if value < 0 {
            return Err(GenApiError::ValueTooWide {
                name: name.to_string(),
                value,
                bit_length,
            });
        }
        let mask = mask_u128(bit_length);
        if (value as u128) > mask {
            return Err(GenApiError::ValueTooWide {
                name: name.to_string(),
                value,
                bit_length,
            });
        }
        Ok(value as u64)
    }
}

fn mask_u128(bit_length: u16) -> u128 {
    if bit_length == 64 {
        u64::MAX as u128
    } else {
        (1u128 << bit_length) - 1
    }
}

fn sign_extend(value: u64, bits: u16) -> i64 {
    let shift = 64 - bits as u32;
    ((value << shift) as i64) >> shift
}

/// Round a floating-point value to i64 using round-to-nearest with ties toward zero.
pub fn round_to_i64(name: &str, value: f64) -> Result<i64, GenApiError> {
    if !value.is_finite() {
        return Err(GenApiError::ExprEval {
            name: name.to_string(),
            msg: "non-finite result".into(),
        });
    }
    let rounded = round_ties_to_zero(value);
    if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return Err(GenApiError::ExprEval {
            name: name.to_string(),
            msg: "result out of range".into(),
        });
    }
    let truncated = rounded.trunc();
    if (rounded - truncated).abs() > 1e-9 {
        return Err(GenApiError::ExprEval {
            name: name.to_string(),
            msg: "unable to represent integer".into(),
        });
    }
    Ok(truncated as i64)
}

fn round_ties_to_zero(value: f64) -> f64 {
    if value >= 0.0 {
        let base = value.floor();
        let frac = value - base;
        if frac > 0.5 { base + 1.0 } else { base }
    } else {
        let base = value.ceil();
        let frac = value - base;
        if frac < -0.5 { base - 1.0 } else { base }
    }
}

/// Apply scale and offset conversion to a raw float register value.
pub fn apply_scale(node: &FloatNode, raw: f64) -> f64 {
    let mut value = raw;
    if let Some((num, den)) = node.scale {
        value *= num as f64 / den as f64;
    }
    if let Some(offset) = node.offset {
        value += offset;
    }
    value
}

/// Encode a user-facing float value back to raw register representation.
pub fn encode_float(node: &FloatNode, value: f64) -> Result<i64, GenApiError> {
    let mut raw = value;
    if let Some(offset) = node.offset {
        raw -= offset;
    }
    if let Some((num, den)) = node.scale {
        if num == 0 {
            return Err(GenApiError::Parse(format!(
                "node {} has zero scale numerator",
                node.name
            )));
        }
        raw *= den as f64 / num as f64;
    }
    let rounded = raw.round();
    if (raw - rounded).abs() > 1e-6 {
        return Err(GenApiError::Range(node.name.clone()));
    }
    let raw_i64 = rounded as i64;
    Ok(raw_i64)
}

/// Decode an IEEE 754 payload into an `f64`.
///
/// `bytes.len()` must be 4 (f32) or 8 (f64). Other widths are rejected because
/// the GenICam Float schema has no other native encodings.
pub fn decode_ieee754(name: &str, bytes: &[u8], order: ByteOrder) -> Result<f64, GenApiError> {
    match bytes.len() {
        4 => {
            let b: [u8; 4] = bytes.try_into().expect("len checked");
            let v = match order {
                ByteOrder::Big => f32::from_be_bytes(b),
                ByteOrder::Little => f32::from_le_bytes(b),
            };
            Ok(v as f64)
        }
        8 => {
            let b: [u8; 8] = bytes.try_into().expect("len checked");
            Ok(match order {
                ByteOrder::Big => f64::from_be_bytes(b),
                ByteOrder::Little => f64::from_le_bytes(b),
            })
        }
        other => Err(GenApiError::Parse(format!(
            "node {name} unsupported IEEE-754 width {other}"
        ))),
    }
}

/// Encode an `f64` as IEEE 754 bytes for a register of width `len` (4 or 8).
///
/// Rejects non-finite inputs and f32 overflow so the caller gets a typed
/// `GenApiError::Range` rather than a silent inf write.
pub fn encode_ieee754(
    name: &str,
    value: f64,
    len: u32,
    order: ByteOrder,
) -> Result<Vec<u8>, GenApiError> {
    match len {
        4 => {
            if !value.is_finite() {
                return Err(GenApiError::Range(name.to_string()));
            }
            if value < f32::MIN as f64 || value > f32::MAX as f64 {
                return Err(GenApiError::Range(name.to_string()));
            }
            let v = value as f32;
            Ok(match order {
                ByteOrder::Big => v.to_be_bytes().to_vec(),
                ByteOrder::Little => v.to_le_bytes().to_vec(),
            })
        }
        8 => {
            if !value.is_finite() {
                return Err(GenApiError::Range(name.to_string()));
            }
            Ok(match order {
                ByteOrder::Big => value.to_be_bytes().to_vec(),
                ByteOrder::Little => value.to_le_bytes().to_vec(),
            })
        }
        other => Err(GenApiError::Parse(format!(
            "node {name} unsupported IEEE-754 width {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_to_i64_decodes_little_endian_when_declared() {
        // The little-endian encoding of 288 (0x0120) in a 4-byte field is
        // [0x20, 0x01, 0x00, 0x00] — this is a real value captured off a
        // Teledyne DALSA Genie Nano's Width register, which declares
        // <Endianess>LittleEndian</Endianess> in its own GenApi XML.
        let bytes = [0x20u8, 0x01, 0x00, 0x00];
        let value = bytes_to_i64("Width", &bytes, Sign::Unsigned, ByteOrder::Little).unwrap();
        assert_eq!(value, 288);
    }

    #[test]
    fn bytes_to_i64_decodes_big_endian_when_declared() {
        let bytes = [0x00u8, 0x00, 0x01, 0x20];
        let value = bytes_to_i64("Width", &bytes, Sign::Unsigned, ByteOrder::Big).unwrap();
        assert_eq!(value, 288);
    }

    #[test]
    fn bytes_to_i64_little_endian_sign_extends_from_the_last_byte() {
        // -1 as a little-endian i32 is [0xFF, 0xFF, 0xFF, 0xFF] either way,
        // so use a value whose sign bit only shows up in the last byte:
        // -2 as little-endian i32 is [0xFE, 0xFF, 0xFF, 0xFF].
        let bytes = [0xFEu8, 0xFF, 0xFF, 0xFF];
        let value = bytes_to_i64("Offset", &bytes, Sign::Signed, ByteOrder::Little).unwrap();
        assert_eq!(value, -2);
    }

    #[test]
    fn i64_to_bytes_round_trips_little_endian() {
        let bytes = i64_to_bytes("Width", 288, 4, Sign::Unsigned, ByteOrder::Little).unwrap();
        assert_eq!(bytes, vec![0x20, 0x01, 0x00, 0x00]);
        let roundtrip = bytes_to_i64("Width", &bytes, Sign::Unsigned, ByteOrder::Little).unwrap();
        assert_eq!(roundtrip, 288);
    }

    #[test]
    fn i64_to_bytes_round_trips_big_endian() {
        let bytes = i64_to_bytes("Width", 288, 4, Sign::Unsigned, ByteOrder::Big).unwrap();
        assert_eq!(bytes, vec![0x00, 0x00, 0x01, 0x20]);
        let roundtrip = bytes_to_i64("Width", &bytes, Sign::Unsigned, ByteOrder::Big).unwrap();
        assert_eq!(roundtrip, 288);
    }
}

/// Map a bitops error to a GenApiError with node context.
pub fn map_bitops_error(name: &str, err: BitOpsError) -> GenApiError {
    match err {
        BitOpsError::UnsupportedWidth { len } => {
            GenApiError::Parse(format!("node {name} uses unsupported register width {len}"))
        }
        BitOpsError::UnsupportedLength { bit_length } => GenApiError::Parse(format!(
            "node {name} uses unsupported bitfield length {bit_length}"
        )),
        BitOpsError::OutOfRange {
            len,
            bit_offset,
            bit_length,
        } => GenApiError::BitfieldOutOfRange {
            name: name.to_string(),
            bit_offset,
            bit_length,
            len,
        },
        BitOpsError::ValueTooWide { bit_length, value } => GenApiError::ValueTooWide {
            name: name.to_string(),
            value: i64::try_from(value).unwrap_or(i64::MAX),
            bit_length,
        },
    }
}

/// Get raw bytes from cache or read from device for read-modify-write operations.
///
/// This helper is used when writing to a bitfield requires first reading the current
/// register value, modifying specific bits, and writing back the result.
pub fn get_raw_or_read(
    cache: &std::cell::RefCell<Option<Vec<u8>>>,
    io: &dyn crate::RegisterIo,
    address: u64,
    len: u32,
) -> Result<Vec<u8>, GenApiError> {
    let cached = cache.borrow().clone();
    if let Some(bytes) = cached
        && bytes.len() == len as usize
    {
        return Ok(bytes);
    }
    io.read(address, len as usize).map_err(|err| match err {
        GenApiError::Io(_) => err,
        other => other,
    })
}
