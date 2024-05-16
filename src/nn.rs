use std::fs::File;

use rand::Rng;

use crate::tools;
use crate::tools::activations;
use crate::tools::matrix::*;
#[allow(dead_code)]
pub struct NN {
    w1: Mat,
    b1: Mat,
    w2: Mat,
    b2: Mat,
    w3: Mat,
    b3: Mat,
    input_shape: usize,
    hidden_layer_size: usize,
    output_shape: usize,
    lr: f64,
}
impl NN {
    pub fn new(input_shape: usize, hidden_layer_size: usize, output_shape: usize) -> NN {
        let w1 = Mat::rand_mat(hidden_layer_size, input_shape, -1.0, 1.0);
        let b1 = Mat::rand_mat(hidden_layer_size, 1, -1.0, 1.0);
        //h1
        let w2 = Mat::rand_mat(hidden_layer_size, hidden_layer_size, -1.0, 1.0);
        let b2 = Mat::rand_mat(hidden_layer_size, 1, -1.0, 1.0);

        let w3 = Mat::rand_mat(output_shape, hidden_layer_size, -1.0, 1.0);
        let b3 = Mat::rand_mat(output_shape, 1, -1.0, 1.0);
        NN {
            w1: w1,
            b1: b1,
            w2: w2,
            b2: b2,
            w3: w3,
            b3: b3,
            input_shape: input_shape,
            hidden_layer_size: hidden_layer_size,
            output_shape: output_shape,
            lr: 0.1,
        }
    }
    pub fn feed_forward(&self, x: &Mat) -> (Mat, Mat, Mat, Mat, Mat, Mat) {
        let z1 = &(&self.w1 * &x) + &self.b1;
        let a1 = z1.map(tools::activations::relu);

        let z2 = &(&self.w2 * &a1) + &self.b2;
        let a2 = z2.map(tools::activations::relu);

        let z3 = &(&self.w3 * &a2) + &self.b3;
        let a3 = z3.map(tools::activations::sigmoid);
        (z1, a1, z2, a2, z3, a3)
    }
    pub fn back_prop(
        &self,
        ins: (&Mat, &Mat, &Mat, &Mat, &Mat, &Mat, &Mat, &Mat),
    ) -> (Mat, Mat, Mat, Mat, Mat, Mat) {
        /*
        //backprop
        let (z1, a1, _z2, a2, z3, a3, x, y) = ins;

        let dz2 = a2 - y;
        // println!("diff:{}", &dz2);
        let dw2 = &dz2 * &a1.transpose();
        let db2 = dz2.clone();
        let dz1 = (&z1.map(activations::drelu)).ele_mul(&(&self.w2.transpose() * &dz2));
        let dw1 = &dz1 * &x.transpose();
        // println!("dz1:{}", &dz1);
        let db1 = dz1.clone();
        (dw1, db1, dw2, db2)
        */
        //backprop
        let (z1, a1, z2, a2, _z3, a3, x, y) = ins;

        let dz3 = a3 - y;
        let dw3 = &dz3 * &a2.transpose();
        let db3 = dz3.clone();

        let dz2 = (&z2.map(activations::drelu)).ele_mul(&(&self.w3.transpose() * &dz3));
        let dw2 = &dz2 * &a1.transpose();
        let db2 = dz2.clone();

        let dz1 = (&z1.map(activations::drelu)).ele_mul(&(&self.w2.transpose() * &dz2));
        let dw1 = &dz1 * &x.transpose();
        let db1 = dz1.clone();

        (dw1, db1, dw2, db2, dw3, db3)
    }
    pub fn set_lr(&mut self, val: f64) {
        self.lr = val;
    }
    pub fn update_param(&mut self, dw1: &Mat, db1: &Mat, dw2: &Mat, db2: &Mat) {
        let lr = self.lr;
        self.w1 = &self.w1 - &dw1.scaler_mul(lr);
        self.b1 = &self.b1 - &db1.scaler_mul(lr);

        self.w2 = &self.w2 - &dw2.scaler_mul(lr);
        self.b2 = &self.b2 - &db2.scaler_mul(lr);
    }
    pub fn train(
        &mut self,
        x: &Vec<Mat>,
        y: &Vec<Mat>,
        epochs: i32,
        batch_size: i32,
        verbose: bool,
    ) {
        let mut epochs_done = 0;

        for _ in 0..epochs {
            let mut loss = Mat::new(self.output_shape, 1);

            for _ in 0..x.len() / batch_size as usize {
                let mut dw1 = Mat::zeroes_like(&self.w1);
                let mut db1 = Mat::zeroes_like(&self.b1);
                let mut dw2 = Mat::zeroes_like(&self.w2);
                let mut db2 = Mat::zeroes_like(&self.b2);
                let mut dw3 = Mat::zeroes_like(&self.w3);
                let mut db3 = Mat::zeroes_like(&self.b3);
                for _ in 0..batch_size {
                    let index = rand::thread_rng().gen_range(0..x.len());

                    let (z1, a1, z2, a2, z3, a3) = self.feed_forward(&x[index]);

                    let (bdw1, bdb1, bdw2, bdb2, bdw3, bdb3) =
                        self.back_prop((&z1, &a1, &z2, &a2, &z3, &a3, &x[index], &y[index]));
                    dw1 = &bdw1 + &dw1;
                    db1 = &bdb1 + &db1;

                    dw2 = &bdw2 + &dw2;
                    db2 = &bdb2 + &db2;

                    dw3 = &bdw3 + &dw3;
                    db3 = &bdb3 + &db3;
                    loss = &loss + (&(&a3 - &y[index]).map(activations::abs));
                }
                dw1 = dw1.scaler_mul(1.0 / batch_size as f64);
                db1 = db1.scaler_mul(1.0 / batch_size as f64);
                dw2 = dw2.scaler_mul(1.0 / batch_size as f64);
                db2 = db2.scaler_mul(1.0 / batch_size as f64);
                self.update_param(&dw1, &db1, &dw2, &db2);
            }
            loss = loss.scaler_mul(1.0 / x.len() as f64);
            let avg_loss = loss.sum_all() / (loss.row() as f64 * loss.col() as f64);
            if verbose {
                println!("epoch:{} loss:{}", epochs_done, avg_loss);
            }
            epochs_done += 1;
        }
    }
}

pub fn parse_mnist(path: &String) -> (Vec<Mat>, Vec<Mat>) {
    let file = File::open(path).expect("couldn't open file ;(");

    let mut rdr = csv::Reader::from_reader(file);

    let mut x = Vec::<Mat>::new();
    let mut y = Vec::<Mat>::new();
    let row = 28;
    let col = 28;
    for result in rdr.records() {
        let record = result.unwrap();

        //one hot encoding
        let mut label = Mat::new(10, 1);
        let index = record.iter().next().unwrap().parse::<usize>().unwrap();
        label[(index, 0)] = 1.0;
        y.push(label);
        let mut buffer = Vec::<f64>::new();
        buffer.reserve(row * col);

        for ele in record.iter().skip(1) {
            buffer.push(ele.parse::<f64>().unwrap());
        }
        let mut ele = Mat::from_vec(buffer, col * row, 1);
        ele.normalize_self();
        ele.scaler_mul(0.9);
        x.push(ele);
    }
    (x, y)
}
