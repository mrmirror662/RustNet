mod tools;

extern crate sdl2;
mod nn;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Rect;

use std::time::Duration;
use tools::matrix::*;
const GRID_SIZE: usize = 28;
const CELL_SIZE: i32 = 20; // Adjust this value to change cell size

fn main() {
    let (x_test, y_test) = nn::parse_mnist(&String::from("mnist_test.csv"));
    let (x_train, y_train) = nn::parse_mnist(&String::from("mnist_train.csv"));
    println!(
        "y size:{}\nx size:{}\ny_train size:{:?}\ny_test size:{:?}",
        y_train.len(),
        x_train.len(),
        y_train[0].shape(),
        x_train[0].shape(),
    );

    let row = 28;
    let col = 28;
    let output_classes = 10;
    let hidden_layer_size = 32;

    let mut nn = nn::NN::new(row * col, hidden_layer_size, output_classes);
    nn.set_lr(0.2);
    nn.train(&x_train, &y_train, 10, 16, true);

    println!("test predictions:");
    let num_prediction = x_test.len() / 2;
    let mut miss = 0;
    for i in 0..num_prediction {
        let prediction = &nn.feed_forward(&x_test[i]).5.get_max().1;
        let label = &y_test[i].get_max().1;
        if prediction != label {
            miss += 1;
        }
    }
    let accuracy = (num_prediction - miss) as f64 / num_prediction as f64;
    println!("accuracy = {}", accuracy);

    let (xo, yo) = nn::parse_mnist(&String::from("output.csv"));
    println!(
        "y size:{}\nx size:{}\ny_train size:{:?}\ny_test size:{:?}",
        yo.len(),
        xo.len(),
        yo[0].shape(),
        xo[0].shape(),
    );

    let prediction = &nn.feed_forward(&xo[0]).5.get_max().1;
    let label = &yo[0].get_max().1;
    println!("(L:{},P:{})", label, prediction);
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window(
            "SDL2 Grid Canvas",
            GRID_SIZE as u32 * CELL_SIZE as u32,
            GRID_SIZE as u32 * CELL_SIZE as u32,
        )
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();

    canvas.set_draw_color(Color::WHITE);
    canvas.clear();
    canvas.present();

    let mut grid: [[bool; GRID_SIZE]; GRID_SIZE] = [[false; GRID_SIZE]; GRID_SIZE];
    let mut is_mouse_down = false;

    let mut event_pump = sdl_context.event_pump().unwrap();

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => {
                    break 'running;
                }
                Event::MouseButtonDown { mouse_btn, .. } => {
                    if mouse_btn == sdl2::mouse::MouseButton::Left {
                        is_mouse_down = true;
                    }
                }
                Event::MouseButtonUp { mouse_btn, .. } => {
                    if mouse_btn == sdl2::mouse::MouseButton::Left {
                        is_mouse_down = false;
                    }
                }
                Event::MouseMotion { x, y, .. } if is_mouse_down => {
                    let cell_x = (x / CELL_SIZE) as usize;
                    let cell_y = (y / CELL_SIZE) as usize;

                    if cell_x < GRID_SIZE && cell_y < GRID_SIZE {
                        grid[cell_y][cell_x] = true;
                        let cell_rect = Rect::new(
                            cell_x as i32 * CELL_SIZE,
                            cell_y as i32 * CELL_SIZE,
                            CELL_SIZE as u32,
                            CELL_SIZE as u32,
                        );
                        canvas.set_draw_color(Color::BLACK);
                        canvas.fill_rect(cell_rect).unwrap();
                        canvas.present();
                    }
                }
                Event::KeyDown {
                    keycode: Some(Keycode::C),
                    ..
                } => {
                    // Clear the grid when 'C' key is pressed
                    grid = [[false; GRID_SIZE]; GRID_SIZE];
                    canvas.set_draw_color(Color::WHITE);
                    canvas.clear();
                    canvas.present();
                }
                _ => {}
            }
        }
        let mut buffer = vec![0.0; 28 * 28];
        for i in 0..28 {
            for j in 0..28 {
                buffer[Mat::map_2_to_1(j, i, col)] = ((grid[i][j] as i32) as f64) * 0.9;
            }
        }
        let x = Mat::from_vec(buffer, row * col, 1);
        let prediction = &nn.feed_forward(&x).5.get_max().1;

        println!("prediction:{}", prediction);
        // Add a small delay to avoid high CPU usage
        std::thread::sleep(Duration::from_millis(10));
    }
    return;
}
