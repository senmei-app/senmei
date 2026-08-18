//! Minimal ONNX weight reader.
//!
//! Parses only the `initializer` tensors (ModelProto -> GraphProto ->
//! TensorProto) from the protobuf wire format, so ONNX-only sources (e.g.
//! Fallin) convert to burnpacks without an ONNX Runtime dependency. The graph
//! is ignored; the initializer names already match the app's state-dict keys.

/// One ONNX initializer tensor. `data` holds the little-endian raw bytes
/// (f16 for `FLOAT16`, f32 for `FLOAT`, ...) as stored in the file.
pub struct OnnxTensor {
    pub name: String,
    pub dims: Vec<i64>,
    pub dtype: i32,
    pub data: Vec<u8>,
}

/// Read all `initializer` tensors from an ONNX model file.
pub fn read_initializers(bytes: &[u8]) -> Result<Vec<OnnxTensor>, String> {
    let graph = first_length_delimited(bytes, 7)
        .ok_or_else(|| "ModelProto has no graph field".to_string())?;
    let mut out = Vec::new();
    for tensor in length_delimited_fields(graph, 5) {
        out.push(parse_tensor(tensor)?);
    }
    Ok(out)
}

fn parse_tensor(msg: &[u8]) -> Result<OnnxTensor, String> {
    let name = first_string(msg, 8)
        .ok_or_else(|| "TensorProto missing name".to_string())?
        .to_string();
    let dtype = first_varint(msg, 2).ok_or_else(|| "TensorProto missing data_type".to_string())? as i32;

    // dims: field 1, packed (length-delimited) or repeated varints.
    let mut dims = Vec::new();
    let mut pos = 0;
    while pos < msg.len() {
        let (tag, next) = varint(msg, pos);
        let (num, wire) = (tag >> 3, tag & 7);
        match (num, wire) {
            (1, 2) => {
                let (len, mut p) = varint(msg, next);
                let end = p + len as usize;
                while p < end {
                    let (v, np) = varint(msg, p);
                    dims.push(v as i64);
                    p = np;
                }
                pos = end;
            }
            (1, 0) => {
                dims.push(varint(msg, next).0 as i64);
                pos = skip(msg, next, wire);
            }
            _ => pos = skip(msg, next, wire),
        }
    }

    // Data: raw_data (field 9), else the packed typed arrays. int32/int64 are
    // varint-packed, so re-encode to little-endian fixed-width.
    let data = if let Some(raw) = first_length_delimited(msg, 9) {
        raw.to_vec()
    } else if let Some(f) = first_length_delimited(msg, 4) {
        f.to_vec() // packed float32
    } else if let Some(d) = first_length_delimited(msg, 10) {
        d.to_vec() // packed double64
    } else if let Some(i) = first_length_delimited(msg, 5) {
        packed_varints_to_bytes(i, 4)
    } else if let Some(i) = first_length_delimited(msg, 7) {
        packed_varints_to_bytes(i, 8)
    } else if let Some(u) = first_length_delimited(msg, 11) {
        packed_varints_to_bytes(u, 8)
    } else {
        Vec::new()
    };

    Ok(OnnxTensor { name, dims, dtype, data })
}

/// Re-encode varint-packed ints as little-endian fixed-width bytes.
fn packed_varints_to_bytes(msg: &[u8], width: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(msg.len() * width);
    let mut pos = 0;
    while pos < msg.len() {
        let (v, next) = varint(msg, pos);
        out.extend_from_slice(&v.to_le_bytes()[..width]);
        pos = next;
    }
    out
}

/// Decode a base-128 varint; returns `(value, next_position)`.
fn varint(b: &[u8], mut pos: usize) -> (u64, usize) {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let byte = b[pos];
        pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (value, pos)
}

/// Skip a field's payload of the given wire type.
fn skip(msg: &[u8], pos: usize, wire: u64) -> usize {
    match wire {
        0 => varint(msg, pos).1,
        1 => pos + 8,
        2 => {
            let (len, next) = varint(msg, pos);
            next + len as usize
        }
        5 => pos + 4,
        _ => pos,
    }
}

/// First length-delimited (wire type 2) field payload with the given number.
fn first_length_delimited<'a>(msg: &'a [u8], want: u64) -> Option<&'a [u8]> {
    length_delimited_fields(msg, want).into_iter().next()
}

/// All length-delimited (wire type 2) field payloads with the given number.
fn length_delimited_fields<'a>(msg: &'a [u8], want: u64) -> Vec<&'a [u8]> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < msg.len() {
        let (tag, next) = varint(msg, pos);
        let (num, wire) = (tag >> 3, tag & 7);
        if num == want && wire == 2 {
            let (len, next) = varint(msg, next);
            out.push(&msg[next..next + len as usize]);
            pos = next + len as usize;
        } else {
            pos = skip(msg, next, wire);
        }
    }
    out
}

/// First varint (wire type 0) value with the given field number.
fn first_varint(msg: &[u8], want: u64) -> Option<u64> {
    let mut pos = 0;
    while pos < msg.len() {
        let (tag, next) = varint(msg, pos);
        let (num, wire) = (tag >> 3, tag & 7);
        if num == want && wire == 0 {
            return Some(varint(msg, next).0);
        }
        pos = skip(msg, next, wire);
    }
    None
}

/// First string (length-delimited UTF-8) payload with the given field number.
fn first_string<'a>(msg: &'a [u8], want: u64) -> Option<&'a str> {
    first_length_delimited(msg, want).and_then(|b| std::str::from_utf8(b).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Serialized ONNX fragment: ModelProto{ graph: GraphProto{ initializer: [
    //   TensorProto{ name:"unet1.conv1.conv.0.weight", dims:[32,12,3,3],
    //     data_type:10 (FLOAT16), raw_data: 4 halfs }
    // ] } }
    fn varint_field(num: u64, wire: u64, value: u64) -> Vec<u8> {
        let tag = (num << 3) | wire;
        let mut out = encode_varint(tag);
        out.extend_from_slice(&encode_varint(value));
        out
    }
    fn len_field(num: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = encode_varint((num << 3) | 2);
        out.extend_from_slice(&encode_varint(payload.len() as u64));
        out.extend_from_slice(payload);
        out
    }
    fn encode_varint(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if v == 0 {
                break;
            }
        }
        out
    }
    fn packed_dims(dims: &[i64]) -> Vec<u8> {
        let mut payload = Vec::new();
        for d in dims {
            payload.extend_from_slice(&encode_varint(*d as u64));
        }
        len_field(1, &payload)
    }
    fn name_field(name: &str) -> Vec<u8> {
        len_field(8, name.as_bytes())
    }
    fn raw_field(data: &[u8]) -> Vec<u8> {
        len_field(9, data)
    }

    #[test]
    fn parses_fp16_initializer() {
        let mut tensor = packed_dims(&[2, 3]);
        tensor.extend_from_slice(&varint_field(2, 0, 10)); // FLOAT16
        tensor.extend_from_slice(&name_field("w.bias"));
        tensor.extend_from_slice(&raw_field(&[0, 0x3c, 0, 0x3c])); // two 1.0 halfs
        let graph = len_field(5, &tensor);
        let model = len_field(7, &graph);

        let tensors = read_initializers(&model).unwrap();
        assert_eq!(tensors.len(), 1);
        let t = &tensors[0];
        assert_eq!(t.name, "w.bias");
        assert_eq!(t.dims, vec![2, 3]);
        assert_eq!(t.dtype, 10);
        assert_eq!(t.data, vec![0, 0x3c, 0, 0x3c]);
    }

    #[test]
    fn skips_unrelated_fields() {
        // ModelProto with ir_version (1) and graph (7); GraphProto with name
        // (2) and one initializer (5); TensorProto with extra field (2 ints).
        let mut t = packed_dims(&[1]);
        t.extend_from_slice(&varint_field(2, 0, 10));
        t.extend_from_slice(&name_field("x.weight"));
        t.extend_from_slice(&varint_field(99, 0, 7)); // unknown field
        t.extend_from_slice(&raw_field(&[0x00, 0x3c]));
        let mut g = len_field(2, b"graph");
        g.extend_from_slice(&len_field(5, &t));
        let mut m = varint_field(1, 0, 8);
        m.extend_from_slice(&len_field(7, &g));

        let tensors = read_initializers(&m).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "x.weight");
    }
}
