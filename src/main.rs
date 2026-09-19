#![feature(iter_array_chunks)]

use std::f32::consts::PI;

use random::Source;
use vgrad::{MeanSquareLoss, MultilayerPerceptron};

type MoonX<const N: usize> = [[f32; 2]; N];
type MoonY<const N: usize> = [[f32; 1]; N];

fn moons_dataset<const SAMPLES: usize>() -> (MoonX<SAMPLES>, MoonY<SAMPLES>) {
    let mut src = random::default(8106493);
    let mut rand = || (src.read_f64() * 0.15) as f32;

    let mut x = [[0f32; 2]; SAMPLES];
    let mut y = [[0f32; 1]; SAMPLES];

    for (i, (px, py)) in
        x.iter_mut().array_chunks::<2>().zip(y.iter_mut().array_chunks::<2>()).enumerate()
    {
        let th = PI * (i as f32) * 2.0 / (SAMPLES as f32);
        let x0 = th.cos() + rand();
        let y0 = th.sin() + rand();
        let x1 = 1.0 - th.cos() + rand();
        let y1 = 1.0 - th.sin() - 0.5 + rand();
        *px[0] = [x0, y0];
        *px[1] = [x1, y1];
        *py[0] = [-1.0f32];
        *py[1] = [1.0f32];
    }

    (x, y)
}

fn main() {
    let (x, y) = moons_dataset::<200>();
    println!("{x:?}");
    println!("{y:?}");

    let mut mlp: MultilayerPerceptron<2, 16, 1, 2> = MultilayerPerceptron::new();
    mlp.randomize(random::default(31337));
    println!("Before: n({:?}) = {:?}, {:?}", x[0], mlp.eval(&x[0]), y[0]);

    let loss = MeanSquareLoss::of(&mut mlp);
    let lr = 0.03;

    for i in 0..10000 {
        let mut total_loss = 0f32;
        for (xs, ys) in x.iter().copied().zip(y.iter().copied()) {
            loss.apply(&mut mlp, xs, ys, lr);
            total_loss += loss.value(&mut mlp);
        }
        println!("Epoch {i:>6}. Loss: {total_loss}");
        println!("    n({:?}) = {:?}, {:?}", x[0], mlp.eval(&x[0]), y[0]);
        println!("    n({:?}) = {:?}, {:?}", x[1], mlp.eval(&x[1]), y[1]);
    }
}
