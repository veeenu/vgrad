#![feature(iter_next_chunk)]

#[derive(Debug, Clone, Copy)]
struct Index(usize);

#[derive(Debug)]
enum Op {
    Sum(Index, Index),
    Mul(Index, Index),
    Tanh(Index),
}

#[derive(Debug)]
struct Value {
    datum: f32,
    grad: f32,
}

impl Value {
    fn new(datum: f32) -> Self {
        Self { datum, grad: 0.0 }
    }
}

#[derive(Debug)]
struct Dag {
    values: Vec<Value>,
    ops: Vec<(Op, Index)>,
}

impl Dag {
    fn new() -> Self {
        Self { values: Vec::new(), ops: Vec::new() }
    }

    fn add_value(&mut self, value: Value) -> Index {
        self.values.push(value);
        Index(self.values.len() - 1)
    }

    fn set_value(&mut self, Index(idx): Index, datum: f32) {
        self.values[idx].datum = datum;
    }

    fn get_value(&self, Index(idx): Index) -> f32 {
        self.values[idx].datum
    }

    fn set_grad(&mut self, Index(idx): Index, grad: f32) {
        self.values[idx].grad = grad;
    }

    fn add_op(&mut self, op: Op) -> Index {
        let out = self.add_value(Value::new(0.0));
        self.ops.push((op, out));
        out
    }

    fn forward(&mut self) {
        for (op, Index(out)) in &self.ops {
            match op {
                Op::Sum(Index(a), Index(b)) => {
                    self.values[*out].datum = self.values[*a].datum + self.values[*b].datum;
                },
                Op::Mul(Index(a), Index(b)) => {
                    self.values[*out].datum = self.values[*a].datum * self.values[*b].datum;
                },
                Op::Tanh(Index(i)) => {
                    self.values[*out].datum = self.values[*i].datum.tanh();
                },
            }
        }
    }

    fn backward(&mut self) {
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
                Op::Tanh(Index(i)) => {
                    self.values[*i].grad +=
                        (1.0 - self.values[*out].datum.powf(2.0)) * self.values[*out].grad;
                },
            }
        }
    }
}

#[derive(Debug)]
struct Perceptron<const N: usize> {
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
}

#[derive(Debug)]
struct Layer<const N: usize, const M: usize> {
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
        (0..N).map(|i| dag.get_value(self.neurons[i].act)).next_chunk().unwrap()
    }
}

#[derive(Debug)]
struct MultilayerPerceptron<const I: usize, const M: usize, const O: usize, const L: usize> {
    dag: Dag,
    input: Layer<I, M>,
    layers: [Layer<M, M>; L],
    output: Layer<M, O>,
}

impl<const I: usize, const M: usize, const O: usize, const L: usize>
    MultilayerPerceptron<I, M, O, L>
{
    fn new() -> Self {
        let mut dag = Dag::new();

        let input = Layer::new_default(&mut dag);

        // TODO didn't feel like MaybeUninit shenanigans but I do want to try and make
        // this fully allocation-free eventually
        let mut layers = Vec::with_capacity(L);

        layers.push(Layer::new(&mut dag, &input));

        for i in 1..L {
            layers.push(Layer::new(&mut dag, &layers[i - 1]));
        }

        let layers = layers.into_iter().next_chunk().unwrap();

        let output = Layer::new(&mut dag, &layers[L - 1]);

        Self { dag, input, layers, output }
    }

    fn eval(&mut self, x: &[f32; I]) -> [f32; O] {
        self.input.eval(&mut self.dag, x);
        self.forward();
        self.output.get(&mut self.dag)
    }

    fn forward(&mut self) {
        self.dag.forward();
    }

    fn backward(&mut self) {
        self.dag.backward();
    }
}

fn main() {
    let mut mlp: MultilayerPerceptron<2, 16, 1, 2> = MultilayerPerceptron::new();
    println!("{:?}", mlp.eval(&[1.0, 1.0]));
    mlp.backward();
}
