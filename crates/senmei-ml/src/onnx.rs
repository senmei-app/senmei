//! Minimal ONNX weight reader.
//!
//! Parses the `initializer` tensors (ModelProto -> GraphProto -> TensorProto)
//! plus any `Constant` node tensors (tensor in the `value` attribute, keyed by
//! the node's output name — some ONNX sources keep weights only in constants).
//! External data (`data_location == EXTERNAL`) is rejected: the sidecar path is
//! relative to the model file and not reachable from a `&[u8]` API. The graph
//! is ignored; the names already match the app's state-dict keys.

/// One ONNX tensor. `data` holds the little-endian raw bytes (f16 for
/// `FLOAT16`, f32 for `FLOAT`, ...) as stored in the file.
#[derive(Debug)]
pub struct OnnxTensor {
    pub name: String,
    pub dims: Vec<i64>,
    pub dtype: i32,
    pub data: Vec<u8>,
}

/// Read all weight tensors (initializers + `Constant` node values) from an
/// ONNX model file. Errors when nothing is found or a tensor uses external data.
pub fn read_initializers(bytes: &[u8]) -> Result<Vec<OnnxTensor>, String> {
    let graph = first_length_delimited(bytes, 7)?
        .ok_or_else(|| "ModelProto has no graph field".to_string())?;
    let mut out = Vec::new();
    for tensor in length_delimited_fields(graph, 5)? {
        let t = parse_tensor(tensor)?;
        // Unnamed initializers are unreachable from the graph — skip them.
        if t.name.is_empty() {
            log::warn!("onnx: skipping unnamed initializer");
            continue;
        }
        out.push(t);
    }
    // Constant nodes: weights can live only in constants. The tensor sits in
    // the `value` attribute; key it by the node's output name (the inner
    // TensorProto.name is usually "value"/empty, so keying by it collides).
    for node in length_delimited_fields(graph, 1)? {
        if first_string(node, 4)? != Some("Constant") {
            continue; // op_type
        }
        let out_name = length_delimited_fields(node, 2)?
            .first()
            .and_then(|b| std::str::from_utf8(b).ok())
            .map(str::to_string);
        for attr in length_delimited_fields(node, 5)? {
            if first_string(attr, 1)? != Some("value") {
                continue; // attribute name
            }
            if first_varint(attr, 20)? != Some(4) {
                continue; // AttributeType::TENSOR
            }
            if let Some(t) = first_length_delimited(attr, 5)? {
                let mut tensor = parse_tensor(t)?;
                if let Some(n) = &out_name {
                    tensor.name = n.clone();
                }
                out.push(tensor);
            }
        }
    }
    if out.is_empty() {
        return Err("no weight tensors found (no initializers or Constant nodes)".to_string());
    }
    Ok(out)
}

fn parse_tensor(msg: &[u8]) -> Result<OnnxTensor, String> {
    // External data is unreachable from a `&[u8]` reader (the sidecar path is
    // relative to the model file) — reject instead of yielding empty data.
    if first_varint(msg, 14)? == Some(1) {
        return Err("TensorProto uses external data, not supported by the byte reader".to_string());
    }
    let name = first_string(msg, 8)?.unwrap_or_default().to_string();
    let dtype = first_varint(msg, 2)?
        .ok_or_else(|| "TensorProto missing data_type".to_string())? as i32;

    // dims: field 1, packed (length-delimited) or repeated varints.
    let mut dims = Vec::new();
    let mut pos = 0;
    while pos < msg.len() {
        let (tag, next) = varint(msg, pos).ok_or_else(|| "malformed dims field".to_string())?;
        let (num, wire) = (tag >> 3, tag & 7);
        match (num, wire) {
            (1, 2) => {
                let (len, mut p) =
                    varint(msg, next).ok_or_else(|| "malformed packed dims".to_string())?;
                let end = p
                    .checked_add(len as usize)
                    .ok_or_else(|| "packed dims overflow".to_string())?;
                if end > msg.len() {
                    return Err("packed dims exceeds buffer".to_string());
                }
                while p < end {
                    let (v, np) = varint(msg, p).ok_or_else(|| "malformed dim".to_string())?;
                    dims.push(v as i64);
                    p = np;
                }
                pos = end;
            }
            (1, 0) => {
                let (v, _) = varint(msg, next).ok_or_else(|| "malformed dim".to_string())?;
                dims.push(v as i64);
                pos = skip(msg, next, wire).ok_or_else(|| "malformed field".to_string())?;
            }
            _ => pos = skip(msg, next, wire).ok_or_else(|| "malformed field".to_string())?,
        }
    }

    // Data: raw_data (field 9), else the packed typed arrays. int32/int64 are
    // varint-packed, so re-encode to little-endian fixed-width.
    let data = if let Some(raw) = first_length_delimited(msg, 9)? {
        raw.to_vec()
    } else if let Some(f) = first_length_delimited(msg, 4)? {
        f.to_vec() // packed float32
    } else if let Some(d) = first_length_delimited(msg, 10)? {
        d.to_vec() // packed double64
    } else if let Some(i) = first_length_delimited(msg, 5)? {
        packed_varints_to_bytes(i, 4)?
    } else if let Some(i) = first_length_delimited(msg, 7)? {
        packed_varints_to_bytes(i, 8)?
    } else if let Some(u) = first_length_delimited(msg, 11)? {
        packed_varints_to_bytes(u, 8)?
    } else {
        Vec::new()
    };

    Ok(OnnxTensor {
        name,
        dims,
        dtype,
        data,
    })
}

/// Re-encode varint-packed ints as little-endian fixed-width bytes.
fn packed_varints_to_bytes(msg: &[u8], width: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(msg.len() * width);
    let mut pos = 0;
    while pos < msg.len() {
        let (v, next) = varint(msg, pos)
            .ok_or_else(|| "malformed varint in packed ints".to_string())?;
        out.extend_from_slice(&v.to_le_bytes()[..width.min(8)]);
        pos = next;
    }
    Ok(out)
}

/// Decode a base-128 varint; returns `(value, next_position)`. `None` when
/// `pos` is out of bounds or the varint overruns the buffer.
fn varint(b: &[u8], mut pos: usize) -> Option<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let byte = *b.get(pos)?;
        pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return None; // overlong varint
        }
    }
    Some((value, pos))
}

/// Skip a field's payload of the given wire type; `None` when malformed.
fn skip(msg: &[u8], pos: usize, wire: u64) -> Option<usize> {
    match wire {
        0 => varint(msg, pos).map(|(_, p)| p),
        1 => pos.checked_add(8).filter(|&p| p <= msg.len()),
        2 => {
            let (len, next) = varint(msg, pos)?;
            next.checked_add(len as usize).filter(|&p| p <= msg.len())
        }
        5 => pos.checked_add(4).filter(|&p| p <= msg.len()),
        _ => Some(pos),
    }
}

/// First length-delimited (wire type 2) field payload with the given number.
fn first_length_delimited<'a>(msg: &'a [u8], want: u64) -> Result<Option<&'a [u8]>, String> {
    Ok(length_delimited_fields(msg, want)?.into_iter().next())
}

/// All length-delimited (wire type 2) field payloads with the given number.
fn length_delimited_fields<'a>(msg: &'a [u8], want: u64) -> Result<Vec<&'a [u8]>, String> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < msg.len() {
        let (tag, next) = varint(msg, pos).ok_or_else(|| "malformed field tag".to_string())?;
        let (num, wire) = (tag >> 3, tag & 7);
        if num == want && wire == 2 {
            let (len, next) =
                varint(msg, next).ok_or_else(|| "malformed field length".to_string())?;
            let end = next
                .checked_add(len as usize)
                .ok_or_else(|| "field length overflow".to_string())?;
            if end > msg.len() {
                return Err("length-delimited field exceeds buffer".to_string());
            }
            out.push(&msg[next..end]);
            pos = end;
        } else {
            pos = skip(msg, next, wire).ok_or_else(|| "malformed field".to_string())?;
        }
    }
    Ok(out)
}

/// First varint (wire type 0) value with the given field number.
fn first_varint(msg: &[u8], want: u64) -> Result<Option<u64>, String> {
    let mut pos = 0;
    while pos < msg.len() {
        let (tag, next) = varint(msg, pos).ok_or_else(|| "malformed field tag".to_string())?;
        let (num, wire) = (tag >> 3, tag & 7);
        if num == want && wire == 0 {
            let (v, _) = varint(msg, next).ok_or_else(|| "malformed varint".to_string())?;
            return Ok(Some(v));
        }
        pos = skip(msg, next, wire).ok_or_else(|| "malformed field".to_string())?;
    }
    Ok(None)
}

/// First string (length-delimited UTF-8) payload with the given field number.
fn first_string<'a>(msg: &'a [u8], want: u64) -> Result<Option<&'a str>, String> {
    Ok(first_length_delimited(msg, want)?.and_then(|b| std::str::from_utf8(b).ok()))
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

    fn constant_node(out_name: &str, tensor: &[u8]) -> Vec<u8> {
        let mut attr = len_field(1, b"value"); // AttributeProto.name
        attr.extend_from_slice(&varint_field(20, 0, 4)); // AttributeProto.type = TENSOR
        attr.extend_from_slice(&len_field(5, tensor)); // AttributeProto.t
        let mut node = len_field(4, b"Constant"); // NodeProto.op_type
        node.extend_from_slice(&len_field(2, out_name.as_bytes())); // NodeProto.output
        node.extend_from_slice(&len_field(5, &attr)); // NodeProto.attribute
        node
    }

    #[test]
    fn reads_constant_node_tensors_keyed_by_output() {
        // Weights only in a Constant node (no initializer) — must be found and
        // keyed by the node's output name, not the inner "value" tensor name.
        let mut t = packed_dims(&[2, 2]);
        t.extend_from_slice(&varint_field(2, 0, 10)); // FLOAT16
        t.extend_from_slice(&name_field("value")); // inner name (ignored)
        t.extend_from_slice(&raw_field(&[0, 0x3c, 0, 0x3c, 0, 0x3c, 0, 0x3c]));
        let node = constant_node("unet.conv.weight", &t);
        let graph = len_field(1, &node); // GraphProto.node
        let model = len_field(7, &graph);

        let tensors = read_initializers(&model).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "unet.conv.weight");
        assert_eq!(tensors[0].dims, vec![2, 2]);
        assert_eq!(tensors[0].data.len(), 8);
    }

    #[test]
    fn merges_initializers_and_constants() {
        let mut init = packed_dims(&[1]);
        init.extend_from_slice(&varint_field(2, 0, 10));
        init.extend_from_slice(&name_field("w1"));
        init.extend_from_slice(&raw_field(&[0, 0x3c]));
        let mut t = packed_dims(&[1]);
        t.extend_from_slice(&varint_field(2, 0, 10));
        t.extend_from_slice(&name_field("value"));
        t.extend_from_slice(&raw_field(&[0, 0x3c]));
        let node = constant_node("w2", &t);
        let mut g = len_field(5, &init); // initializer
        g.extend_from_slice(&len_field(1, &node)); // node
        let model = len_field(7, &g);

        let tensors = read_initializers(&model).unwrap();
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].name, "w1");
        assert_eq!(tensors[1].name, "w2");
    }

    #[test]
    fn rejects_external_data() {
        // data_location (field 14) = EXTERNAL (1): unreachable from bytes.
        let mut t = packed_dims(&[1]);
        t.extend_from_slice(&varint_field(2, 0, 10));
        t.extend_from_slice(&name_field("w"));
        t.extend_from_slice(&varint_field(14, 0, 1)); // EXTERNAL
        let graph = len_field(5, &t);
        let model = len_field(7, &graph);

        let err = read_initializers(&model).unwrap_err();
        assert!(err.contains("external"), "unexpected: {err}");
    }

    #[test]
    fn errors_when_no_weights_found() {
        // Graph with no initializers and only a non-Constant node.
        let node = len_field(4, b"Conv");
        let graph = len_field(1, &node);
        let model = len_field(7, &graph);

        let err = read_initializers(&model).unwrap_err();
        assert!(err.contains("no weight tensors"), "unexpected: {err}");
    }

    #[test]
    fn errors_on_truncated_length_delimited_field() {
        // initializer field (5) claims 100 bytes but only 2 are present.
        let mut bad = encode_varint((5 << 3) | 2);
        bad.extend_from_slice(&encode_varint(100));
        bad.extend_from_slice(&[0u8; 2]);
        let model = len_field(7, &bad);

        let err = read_initializers(&model).unwrap_err();
        assert!(err.contains("exceeds buffer"), "unexpected: {err}");
    }

    #[test]
    fn errors_on_truncated_varint() {
        // initializer length varint has the continuation bit set, then EOF.
        let mut bad = encode_varint((5 << 3) | 2);
        bad.push(0x80);
        let model = len_field(7, &bad);

        let err = read_initializers(&model).unwrap_err();
        assert!(err.contains("malformed"), "unexpected: {err}");
    }
}
