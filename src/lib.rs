#![feature(iter_next_chunk)]

use std::iter;
use std::mem::MaybeUninit;

#[derive(Debug, Clone, Copy)]
pub struct Index(usize);

#[derive(Debug)]
pub enum Op {
    Sum(Index, Index),
    Mul(Index, Index),
    Neg(Index),
    Tanh(Index),
}

#[derive(Debug)]
pub struct Value {
    datum: f32,
    grad: f32,
}

impl Value {
    fn new(datum: f32) -> Self {
        Self { datum, grad: 0.0 }
    }
}

#[derive(Debug, Default)]
pub struct Dag {
    values: Vec<Value>,
    ops: Vec<(Op, Index)>,
}

impl Dag {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn add_value(&mut self, value: Value) -> Index {
        self.values.push(value);
        Index(self.values.len() - 1)
    }

    pub fn set_value(&mut self, Index(idx): Index, datum: f32) {
        self.values[idx].datum = datum;
    }

    pub fn get_value(&self, Index(idx): Index) -> f32 {
        self.values[idx].datum
    }

    pub fn set_grad(&mut self, Index(idx): Index, grad: f32) {
        self.values[idx].grad = grad;
    }

    pub fn get_grad(&mut self, Index(idx): Index) -> f32 {
        self.values[idx].grad
    }

    pub fn zero_grad(&mut self) {
        for val in &mut self.values {
            val.grad = 0.0;
        }
    }

    pub fn add_op(&mut self, op: Op) -> Index {
        let out = self.add_value(Value::new(0.0));
        self.ops.push((op, out));
        out
    }

    pub fn forward(&mut self) {
        for (op, Index(out)) in &self.ops {
            match op {
                Op::Sum(Index(a), Index(b)) => {
                    self.values[*out].datum = self.values[*a].datum + self.values[*b].datum;
                },
                Op::Mul(Index(a), Index(b)) => {
                    self.values[*out].datum = self.values[*a].datum * self.values[*b].datum;
                },
                Op::Neg(Index(i)) => {
                    self.values[*out].datum = -self.values[*i].datum;
                },
                Op::Tanh(Index(i)) => {
                    self.values[*out].datum = self.values[*i].datum.tanh();
                },
            }
        }
    }

    pub fn backward(&mut self) {
        for (op, Index(out)) in self.ops.iter().rev() {
            match op {
                Op::Sum(Index(a), Index(b)) => {
                    self.values[*a].grad += self.values[*out].grad;
                    self.values[*b].grad += self.values[*out].grad;
                },
                Op::Mul(Index(a), Index(b)) => {
                    self.values[*a].grad += self.values[*b].datum * self.values[*out].grad;
                    self.values[*b].grad += self.values[*a].datum * self.values[*out].grad;
                },
                Op::Neg(Index(i)) => {
                    self.values[*i].grad -= self.values[*out].grad;
                },
                Op::Tanh(Index(i)) => {
                    self.values[*i].grad +=
                        (1.0 - self.values[*out].datum.powf(2.0)) * self.values[*out].grad;
                },
            }
        }
    }

    pub fn new_layer<const P: usize, const N: usize, const M: usize>(
        &mut self,
        input: &Layer<P, N>,
    ) -> Layer<N, M> {
        Layer::new(self, input)
    }
}

#[derive(Debug)]
pub struct Perceptron<const N: usize> {
    x: [Index; N],
    w: [Index; N],
    b: Index,
    muls: [Index; N],
    sums: [Index; N],
    act: Index,
}

impl<const N: usize> Perceptron<N> {
    pub fn new(dag: &mut Dag, x: [Index; N]) -> Self {
        let mut w = [Index(0); N];
        let mut muls = [Index(0); N];
        let mut sums = [Index(0); N];

        for w in w.iter_mut() {
            *w = dag.add_value(Value::new(0.0));
        }
        let b = dag.add_value(Value::new(0.0));

        for i in 0..N {
            muls[i] = dag.add_op(Op::Mul(x[i], w[i]));
        }

        sums[0] = dag.add_op(Op::Sum(muls[0], muls[1]));

        for i in 1..N - 1 {
            sums[i] = dag.add_op(Op::Sum(sums[i - 1], muls[i + 1]));
        }

        sums[N - 1] = dag.add_op(Op::Sum(sums[N - 2], b));

        let act = dag.add_op(Op::Tanh(sums[N - 1]));

        Self { x, w, b, muls, sums, act }
    }

    pub fn new_default(dag: &mut Dag) -> Self {
        let mut x = [Index(0); N];

        for x in x.iter_mut() {
            *x = dag.add_value(Value::new(rand::random()));
        }

        Self::new(dag, x)
    }

    pub fn eval(&self, dag: &mut Dag, x: &[f32; N]) {
        for (x, ix) in self.x.iter().zip(x.iter()) {
            dag.set_value(*x, *ix);
        }
    }

    pub fn output(&self) -> Index {
        self.act
    }

    fn params(&self) -> impl Iterator<Item = Index> {
        self.w.iter().copied().chain(iter::once(self.b))
    }
}

#[derive(Debug)]
pub struct Layer<const N: usize, const M: usize> {
    neurons: [Perceptron<N>; M],
}

impl<const N: usize, const M: usize> Layer<N, M> {
    pub fn new<const P: usize>(dag: &mut Dag, input: &Layer<P, N>) -> Self {
        let neurons = (0..M)
            .map(|_| {
                Perceptron::new(dag, input.neurons.iter().map(|n| n.output()).next_chunk().unwrap())
            })
            .next_chunk()
            .unwrap();

        Self { neurons }
    }

    pub fn new_default(dag: &mut Dag) -> Self {
        let neurons = (0..M).map(|_| Perceptron::new_default(dag)).next_chunk().unwrap();

        Self { neurons }
    }

    pub fn eval(&mut self, dag: &mut Dag, x: &[f32; N]) {
        for neuron in &self.neurons {
            neuron.eval(dag, x);
        }
    }

    pub fn get(&mut self, dag: &mut Dag) -> [f32; M] {
        (0..M).map(|i| dag.get_value(self.neurons[i].act)).next_chunk().unwrap()
    }

    fn params(&self) -> impl Iterator<Item = Index> {
        self.neurons.iter().flat_map(Perceptron::params)
    }
}

#[derive(Debug)]
pub struct MultilayerPerceptron<const I: usize, const M: usize, const O: usize, const L: usize> {
    dag: Dag,
    input: Layer<I, M>,
    layers: [Layer<M, M>; L],
    output: Layer<M, O>,
}

impl<const I: usize, const M: usize, const O: usize, const L: usize>
    MultilayerPerceptron<I, M, O, L>
{
    pub fn new() -> Self {
        let mut dag = Dag::new();

        let input = Layer::new_default(&mut dag);

        let mut layers = [const { MaybeUninit::uninit() }; L];

        layers[0].write(dag.new_layer(&input));

        for i in 1..L {
            layers[i].write(dag.new_layer(unsafe { layers[i - 1].assume_init_ref() }));
        }

        let layers = layers.into_iter().map(|m| unsafe { m.assume_init() }).next_chunk().unwrap();

        let output = dag.new_layer(&layers[L - 1]);

        Self { dag, input, layers, output }
    }

    pub fn randomize<S: random::Source>(&mut self, mut src: S) {
        let Self { dag, input, layers, output } = self;
        let params =
            input.params().chain(layers.iter().flat_map(Layer::params)).chain(output.params());
        for param in params {
            dag.set_value(param, (src.read_f64() * 2.0 - 1.0) as f32);
        }
    }

    pub fn eval(&mut self, x: &[f32; I]) -> [f32; O] {
        self.input.eval(&mut self.dag, x);
        self.forward();
        self.output.get(&mut self.dag)
    }

    pub fn forward(&mut self) {
        self.dag.forward();
    }

    pub fn backward(&mut self) {
        self.dag.backward();
    }

    fn learn(&mut self, lr: f32) {
        let Self { dag, input, layers, output } = self;
        let params =
            input.params().chain(layers.iter().flat_map(Layer::params).chain(output.params()));

        for param in params {
            let w = dag.get_value(param);
            let g = dag.get_grad(param);
            dag.set_value(param, w - g * lr);
        }
    }
}

impl<const I: usize, const M: usize, const O: usize, const L: usize> Default
    for MultilayerPerceptron<I, M, O, L>
{
    fn default() -> Self {
        Self::new()
    }
}

pub struct MeanSquareLoss<const O: usize> {
    pred: [Index; O],
    loss: Index,
}

impl<const O: usize> MeanSquareLoss<O> {
    pub fn of<const I: usize, const M: usize, const L: usize>(
        mlp: &mut MultilayerPerceptron<I, M, O, L>,
    ) -> Self {
        let mut pred = [Index(0); O];

        for pred in &mut pred {
            *pred = mlp.dag.add_value(Value::new(0.0));
        }

        let mut it = pred.iter().copied().zip(mlp.output.neurons.iter().map(|p| p.output()));
        let mut loss = {
            let (pred, out) = it.next().unwrap();
            let pred_neg = mlp.dag.add_op(Op::Neg(pred));
            let diff = mlp.dag.add_op(Op::Sum(pred_neg, out));
            mlp.dag.add_op(Op::Mul(diff, diff))
        };

        for (pred, out) in it {
            let pred_neg = mlp.dag.add_op(Op::Neg(pred));
            let diff = mlp.dag.add_op(Op::Sum(pred_neg, out));
            let sq_diff = mlp.dag.add_op(Op::Mul(diff, diff));
            loss = mlp.dag.add_op(Op::Sum(loss, sq_diff));
        }

        Self { pred, loss }
    }

    pub fn apply<const I: usize, const M: usize, const L: usize>(
        &self,
        mlp: &mut MultilayerPerceptron<I, M, O, L>,
        x: [f32; I],
        y: [f32; O],
        lr: f32,
    ) {
        mlp.dag.zero_grad();
        mlp.dag.set_grad(self.loss, 1.0);
        for (index, y) in self.pred.iter().copied().zip(y.iter().copied()) {
            mlp.dag.set_value(index, y);
        }
        mlp.eval(&x);
        mlp.dag.backward();

        mlp.learn(lr);
    }

    pub fn value<const I: usize, const M: usize, const L: usize>(
        &self,
        mlp: &mut MultilayerPerceptron<I, M, O, L>,
    ) -> f32 {
        mlp.dag.get_value(self.loss)
    }
}
