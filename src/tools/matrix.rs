use core::fmt;

use rand::Rng;
use std::ops::{Add, Index, IndexMut, Mul, Sub};

// Define your struct
#[derive(Clone)]
pub struct Mat {
    buffer: Vec<f64>,
    size: usize,
    row: usize,
    col: usize,
}

// Implement methods for the struct
#[allow(dead_code)]
impl Mat {
    // Constructor for a new matrix
    pub fn new(row: usize, col: usize) -> Mat {
        let buffer = vec![0.0; row * col];
        let size = row * col;
        Mat {
            buffer,
            size,
            row,
            col,
        }
    }
    pub fn val_mat(row: usize, col: usize, val: f64) -> Mat {
        let mut buffer: Vec<f64> = vec![0.0; row * col];
        for i in 0..row * col {
            buffer[i] = val
        }
        Mat {
            buffer,
            size: row * col,
            row,
            col,
        }
    }
    // Constructor for a matrix with random values
    pub fn rand_mat(row: usize, col: usize, min: f64, max: f64) -> Mat {
        let mut buffer: Vec<f64> = vec![0.0; row * col];
        for i in 0..row * col {
            buffer[i] = rand::thread_rng().gen_range(min..max);
        }
        Mat {
            buffer,
            size: row * col,
            row,
            col,
        }
    }
    pub fn zeroes_like(other: &Mat) -> Mat {
        let row = other.row();
        let col = other.col();
        let buffer: Vec<f64> = vec![0.0; row * col];

        Mat {
            buffer,
            size: row * col,
            row,
            col,
        }
    }
    pub fn from_vec(buffer: Vec<f64>, row: usize, col: usize) -> Mat {
        Mat {
            buffer,
            size: row * col,
            row,
            col,
        }
    }
    pub fn row(&self) -> usize {
        self.row
    }
    pub fn col(&self) -> usize {
        self.col
    }
    // Function to map 2D index to 1D index
    pub fn map_2_to_1(x: usize, y: usize, width: usize) -> usize {
        width * y + x
    }

    // Display function to print the matrix
    pub fn display(&self) {
        for i in 0..self.row {
            for j in 0..self.col {
                print!("{},", self.buffer[Mat::map_2_to_1(j, i, self.col)]);
            }
            println!();
        }
    }

    // Getter function to get the index in the buffer for given coordinates
    fn get(&self, x: usize, y: usize) -> usize {
        assert!(y < self.row && x < self.col);
        Mat::map_2_to_1(x, y, self.col)
    }

    // Setter function to set value at given coordinates
    pub fn set(&mut self, x: usize, y: usize, val: f64) {
        assert!(y < self.row && x < self.col);
        self.buffer[Mat::map_2_to_1(x, y, self.col)] = val;
    }

    // Transpose function to transpose the matrix
    pub fn transpose(&self) -> Mat {
        let mut mat_t = Mat::new(self.col, self.row);

        for i in 0..self.row {
            for j in 0..self.col {
                mat_t[(j, i)] = self[(i, j)];
            }
        }
        mat_t
    }

    //map each element to itself through a function
    pub fn map(&self, f: fn(f64) -> f64) -> Mat {
        let mut mat = self.clone();
        for i in 0..self.buffer.len() {
            mat.buffer[i] = f(self.buffer[i]);
        }
        mat
    }
    pub fn sum(&self, row: usize) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.col {
            sum += self[(row, i)];
        }
        sum
    }
    pub fn ele_mul(&self, other: &Mat) -> Mat {
        assert_eq!(self.col, other.col);
        assert_eq!(self.row, other.row);
        let mut buffer = self.buffer.clone();
        for i in 0..self.size {
            buffer[i] *= other.buffer[i];
        }
        Mat {
            buffer,
            row: self.row,
            col: self.col,
            size: self.size,
        }
    }
    pub fn scaler_mul(&self, val: f64) -> Mat {
        let mut buffer = self.buffer.clone();

        for i in 0..self.size {
            buffer[i] *= val;
        }
        Mat {
            buffer,
            row: self.row,
            col: self.col,
            size: self.size,
        }
    }
    pub fn shape(&self) -> (usize, usize) {
        {
            (self.row, self.col)
        }
    }
    pub fn normalize_self(&mut self) {
        let mut max = libm::fabs(self[(0, 0)]);
        for ele in &self.buffer {
            if max < libm::fabs(*ele) {
                max = *ele;
            }
        }
        for i in 0..self.buffer.len() {
            self.buffer[i] /= max;
        }
    }
    pub fn get_max(&self) -> (usize, usize) {
        let mut max = libm::fabs(self[(0, 0)]);
        let mut max_index = 0;
        for ele in self.buffer.iter().enumerate() {
            if max < libm::fabs(*ele.1) {
                max = *ele.1;
                max_index = ele.0;
            }
        }
        (max_index % self.col, max_index / self.col)
    }
    pub fn sum_all(&self) -> f64 {
        let mut sum = 0.0;
        for ele in &self.buffer {
            sum += ele;
        }
        sum
    }
}

// Implement indexing for the struct
impl Index<(usize, usize)> for Mat {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        &self.buffer[Mat::map_2_to_1(j, i, self.col)]
    }
}

// Implement mutable indexing for the struct
impl IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        assert!(i < self.row);
        assert!(j < self.col);
        &mut self.buffer[Mat::map_2_to_1(j, i, self.col)]
    }
}

// Implement addition for the struct
impl Add for &Mat {
    type Output = Mat;

    fn add(self, other: &Mat) -> Mat {
        assert_eq!(self.col, other.col);
        assert_eq!(self.row, other.row);
        let mut buffer = self.buffer.clone();
        for i in 0..self.size {
            buffer[i] += other.buffer[i];
        }
        Mat {
            buffer,
            row: self.row,
            col: self.col,
            size: self.size,
        }
    }
}

// Implement subtraction for the struct
impl Sub for &Mat {
    type Output = Mat;

    fn sub(self, other: &Mat) -> Mat {
        assert_eq!(self.col, other.col);
        assert_eq!(self.row, other.row);
        let mut buffer = self.buffer.clone();
        for i in 0..self.size {
            buffer[i] -= other.buffer[i];
        }
        Mat {
            buffer,
            row: self.row,
            col: self.col,
            size: self.size,
        }
    }
}

// Implement multiplication for the struct
impl Mul for &Mat {
    type Output = Mat;
    fn mul(self, other: &Mat) -> Mat {
        /*
        let mut a = faer::Mat::<f64>::zeros(self.row, self.col);
        let mut b = faer::Mat::<f64>::zeros(other.row, other.col);

        for i in 0..self.row {
            for j in 0..self.col {
                unsafe {
                    a.write_unchecked(i, j, self[(i, j)]);
                }
            }
        }
        for i in 0..other.row {
            for j in 0..other.col {
                unsafe {
                    b.write_unchecked(i, j, other[(i, j)]);
                }
            }
        }
        let mut c = faer::Mat::<f64>::zeros(self.row, other.col);

        // Computes `faer::scale(3.0) * &a * &b` and stores the result in `c`.
        matmul(
            c.as_mut(),
            a.as_ref(),
            b.as_ref(),
            None,
            1.0,
            faer::Parallelism::Rayon(0),
        );

        let mut mat_result = Mat::new(self.row, other.col);
        for i in 0..mat_result.row {
            for j in 0..mat_result.col {
                unsafe {
                    mat_result[(i, j)] = c.read_unchecked(i, j);
                }
            }
        }
        mat_result
        */

        assert_eq!(self.col, other.row);
        let other_transposed = other.transpose();
        let mut mat_result = Mat::new(self.row, other.col);

        let block_size = 32; // Adjust the block size based on your cache size and architecture

        for i in (0..self.row).step_by(block_size) {
            for j in (0..other_transposed.row).step_by(block_size) {
                for k in (0..self.col).step_by(block_size) {
                    for ii in i..(i + block_size).min(self.row) {
                        for jj in j..(j + block_size).min(other_transposed.row) {
                            for kk in k..(k + block_size).min(self.col) {
                                mat_result[(ii, jj)] += self[(ii, kk)] * other_transposed[(jj, kk)];
                            }
                        }
                    }
                }
            }
        }
        mat_result
    }
}

// Implement display for the struct
impl fmt::Display for &Mat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.row {
            for j in 0..self.col {
                write!(f, "{},", self.buffer[Mat::map_2_to_1(j, i, self.col)])
                    .expect("idk dawg this printin error");
            }
            write!(f, "\n").expect("idk dawg this printin error");
        }
        write!(f, "\n")
    }
}
