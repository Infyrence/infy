use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ============================================================================
// MODULE 1: Fast JSON parsing + partial JSON for streaming
// ============================================================================

#[pyfunction]
fn parse_json(py: Python<'_>, text: &str) -> PyResult<PyObject> {
    match serde_json::from_str::<Value>(text) {
        Ok(val) => Ok(json_to_pyobject(py, &val)?),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "JSON parse error: {e}"
        )))
    }
}

#[pyfunction]
fn parse_partial_json(py: Python<'_>, text: &str) -> PyResult<PyObject> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(py.None());
    }

    if let Ok(val) = serde_json::from_str::<Value>(text) {
        return Ok(json_to_pyobject(py, &val)?);
    }

    let bytes = text.as_bytes();
    for i in (0..bytes.len()).rev() {
        if bytes[i] == b'}' || bytes[i] == b']' || bytes[i] == b'"' {
            if let Ok(val) = serde_json::from_str(&text[..=i]) {
                return Ok(json_to_pyobject(py, &val)?);
            }
        }
    }

    Ok(py.None())
}

#[pyfunction]
fn extract_json_from_markdown(py: Python<'_>, text: &str) -> PyResult<PyObject> {
    if let Some(body) = find_fenced_block(text) {
        if let Ok(val) = serde_json::from_str::<Value>(body.trim()) {
            return Ok(json_to_pyobject(py, &val)?);
        }
    }
    Ok(py.None())
}

/// Extract the contents of the first fenced code block, tolerating an optional
/// language tag (```json, ```JSON, ``` ...) and surrounding whitespace. This
/// mirrors the Python fallback regex so both code paths behave identically.
fn find_fenced_block(text: &str) -> Option<&str> {
    let start_fence = text.find("```")?;
    let after_fence = &text[start_fence + 3..];
    // Skip an optional language hint up to the first newline.
    let newline = after_fence.find('\n')?;
    let body = &after_fence[newline + 1..];
    // Capture everything up to the closing fence.
    let end = body.find("```")?;
    Some(&body[..end])
}

// ============================================================================
// MODULE 2: Token counting (fast approximation)
// ============================================================================

#[pyfunction]
fn count_tokens(text: &str) -> usize {
    let mut count = 0;
    let mut chars = text.char_indices().peekable();

    while let Some((i, c)) = chars.next() {
        if c.is_whitespace() {
            continue;
        }

        if c.is_ascii_alphanumeric() || c == '_' {
            let start = i;
            while let Some(&(_, nc)) = chars.peek() {
                if nc.is_ascii_alphanumeric() || nc == '_' {
                    chars.next();
                } else {
                    break;
                }
            }
            let word_len = i - start + 1;
            count += (word_len + 3) / 4;
        } else if c.is_ascii_punctuation() {
            count += 1;
        } else if is_cjk(c) {
            while let Some(&(_, nc)) = chars.peek() {
                if is_cjk(nc) {
                    chars.next();
                } else {
                    break;
                }
            }
            count += 1;
        } else {
            count += 1;
        }
    }

    count.max(1)
}

fn is_cjk(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x3000..=0x303F).contains(&cp)
        || (0x3040..=0x309F).contains(&cp)
        || (0x30A0..=0x30FF).contains(&cp)
        || (0xFF00..=0xFFEF).contains(&cp)
}

#[pyfunction]
fn count_tokens_batch(texts: Vec<String>) -> Vec<usize> {
    texts.iter().map(|t| count_tokens(t)).collect()
}

// ============================================================================
// MODULE 3: SIMD-accelerated cosine similarity
// ============================================================================

#[pyfunction]
fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Vector length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Ok(0.0);
    }
    Ok(cosine_from_sums(dot_and_norms(&a, &b)))
}

fn cosine_from_sums((dot, norm_a, norm_b): (f32, f32, f32)) -> f32 {
    let denom = (norm_a as f64 * norm_b as f64).sqrt() as f32;
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

/// Compute (dot, ||a||^2, ||b||^2). Dispatches to an AVX2+FMA kernel only after
/// confirming the CPU supports it at runtime; otherwise uses a scalar fallback.
/// This avoids SIGILL on x86_64 chips without AVX2 and is correct on every arch.
fn dot_and_norms(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature checks above.
            return unsafe { dot_and_norms_avx2(a, b) };
        }
    }
    dot_and_norms_scalar(a, b)
}

fn dot_and_norms_scalar(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    (dot, norm_a, norm_b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_and_norms_avx2(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;

    let mut acc_dot = _mm256_setzero_ps();
    let mut acc_a = _mm256_setzero_ps();
    let mut acc_b = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        acc_dot = _mm256_fmadd_ps(va, vb, acc_dot);
        acc_a = _mm256_fmadd_ps(va, va, acc_a);
        acc_b = _mm256_fmadd_ps(vb, vb, acc_b);
    }

    let dot_arr: [f32; 8] = std::mem::transmute(acc_dot);
    let a_arr: [f32; 8] = std::mem::transmute(acc_a);
    let b_arr: [f32; 8] = std::mem::transmute(acc_b);
    let mut dot: f32 = dot_arr.iter().sum();
    let mut norm_a: f32 = a_arr.iter().sum();
    let mut norm_b: f32 = b_arr.iter().sum();

    // Handle the tail elements that don't fill a full 8-lane vector — exactly
    // once. (The previous version added these on top of a full scalar pass on
    // non-x86 builds, double-counting them.)
    for i in (chunks * 8)..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    (dot, norm_a, norm_b)
}

#[pyfunction]
fn batch_cosine_similarity(
    _py: Python<'_>,
    query: Vec<f32>,
    documents: Vec<Vec<f32>>,
    top_k: Option<usize>,
) -> PyResult<Vec<(usize, f32)>> {
    let k = top_k.unwrap_or(documents.len());
    let mut results: Vec<(usize, f32)> = Vec::with_capacity(documents.len());

    for (i, doc) in documents.iter().enumerate() {
        let score = cosine_similarity(query.clone(), doc.clone())?;
        results.push((i, score));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    Ok(results)
}

#[pyfunction]
fn inner_product(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Vector length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

// ============================================================================
// MODULE 4: Hashing / fingerprinting
// ============================================================================

#[pyfunction]
fn fast_hash(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

#[pyfunction]
fn fast_hash_batch(texts: Vec<String>) -> Vec<u64> {
    texts.iter().map(|t| fast_hash(t)).collect()
}

// ============================================================================
// Helper: JSON → Python object conversion
// ============================================================================

fn json_to_pyobject(py: Python<'_>, val: &Value) -> PyResult<PyObject> {
    match val {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => {
            let any: PyObject = if *b {
                true.into_pyobject(py)?.to_owned().into_any().unbind()
            } else {
                false.into_pyobject(py)?.to_owned().into_any().unbind()
            };
            Ok(any)
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(n.to_string().into_pyobject(py)?.into_any().unbind())
            }
        }
        Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_pyobject(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k.as_str(), json_to_pyobject(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

// ============================================================================
// Module definition
// ============================================================================

#[pymodule]
fn infy_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_json, m)?)?;
    m.add_function(wrap_pyfunction!(parse_partial_json, m)?)?;
    m.add_function(wrap_pyfunction!(extract_json_from_markdown, m)?)?;
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(count_tokens_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(batch_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(inner_product, m)?)?;
    m.add_function(wrap_pyfunction!(fast_hash, m)?)?;
    m.add_function(wrap_pyfunction!(fast_hash_batch, m)?)?;
    Ok(())
}
