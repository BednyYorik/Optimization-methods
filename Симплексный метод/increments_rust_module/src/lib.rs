use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn find_increments(n:f64, m:f64) -> (f64, f64) {
    let b1:f64 ;
    let b2:f64 ;

    b1 = (((n+1.0_f64).sqrt() - 1.0_f64) / (n * (2.0_f64).sqrt())) * m;
    b2 = (((n+1.0_f64).sqrt() + n - 1.0_f64) / (n * (2.0_f64).sqrt()))*m;

    println!("Приращения: betta1 = {}, betta2 = {}", b1, b2);
    return (b1, b2)
}

/// A Python module implemented in Rust.
#[pymodule]
fn increments_rust_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_increments, m)?)?;
    Ok(())
}