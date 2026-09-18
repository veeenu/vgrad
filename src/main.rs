use vgrad::MultilayerPerceptron;

fn main() {
    let mut mlp: MultilayerPerceptron<2, 16, 1, 2> = MultilayerPerceptron::new();
    println!("{:?}", mlp.eval(&[1.0, 1.0]));
    mlp.backward();
}
