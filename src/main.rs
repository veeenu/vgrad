use std::f32::consts::PI;

use random::Source;
use vgrad::MultilayerPerceptron;

type MoonX<const N: usize> = [((f32, f32), (f32, f32)); N];
type MoonY<const N: usize> = [(f32, f32); N];

fn moons_dataset<const SAMPLES: usize>() -> (MoonX<SAMPLES>, MoonY<SAMPLES>) {
    let mut src = random::default(8106493);
    let mut rand = || (src.read_f64() * 0.15) as f32;

    let mut x = [((0f32, 0f32), (0f32, 0f32)); SAMPLES];

    for (i, points) in x.iter_mut().enumerate() {
        let th = PI * (i as f32) / (SAMPLES as f32);
        let x0 = th.cos() + rand();
        let y0 = th.sin() + rand();
        let x1 = 1.0 - th.cos() + rand();
        let y1 = 1.0 - th.sin() - 0.5 + rand();
        *points = ((x0, y0), (x1, y1));
    }

    let y = [(0f32, 0f32); SAMPLES];

    (x, y)
}

fn main() {
    let dataset = moons_dataset::<200>();
    println!("{:?}", dataset);

    let mut mlp: MultilayerPerceptron<2, 16, 1, 2> = MultilayerPerceptron::new();
    println!("{:?}", mlp.eval(&[1.0, 1.0]));
    mlp.backward();
}
