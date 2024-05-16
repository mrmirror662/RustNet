use libm::*;

pub fn sigmoid(x: f64) -> f64 {
    return exp(x) / (1.0 + exp(x));
}
pub fn dsigmoid(x: f64) -> f64 {
    return x * (1.0 - x);
}
pub fn relu(x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    return x;
}
pub fn drelu(x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    return 1.0;
}

pub fn tanh(x: f64) -> f64 {
    return libm::tanh(x);
}
pub fn dtanh(x: f64) -> f64 {
    return 1.0 - (x * x);
}

pub fn abs(x: f64) -> f64 {
    if x < 0.0 {
        return -1.0 * x;
    }
    return x;
}
