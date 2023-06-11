use itertools::Itertools;
use rand::prelude::*;

const TIMELIMIT: f64 = 1.85;

fn main() {
    let mut timer = Timer::new();
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let input = read_input();
    let mut output = greedy(&input, &mut rng);
    // eprintln!("{}", compute_score(&input, &output));
    annealing(&input, &mut output, &mut timer, &mut rng);
    output.write();
    // eprintln!("{}", compute_score(&input, &output));
}

fn greedy(input: &Input, rng: &mut rand_chacha::ChaCha20Rng) -> Output {
    let p = (0..input.n).map(|_| rng.gen_range(100, 5000)).collect_vec();
    let mut b = vec![false; input.m];
    // クラスカル
    let sorted_edges = {
        let mut edges = input
            .edges
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, e)| (i, e))
            .collect::<Vec<_>>();
        edges.sort_by_key(|(_, e)| e.2);
        edges
    };
    let mut dsu = Dsu::new(input.n);
    for (i, (u, v, _)) in sorted_edges {
        if !dsu.same(u, v) {
            dsu.merge(u, v);
            b[i] = true;
        }
    }
    Output {
        powers: p,
        edges: b,
    }
}

fn annealing(
    input: &Input,
    output: &mut Output,
    timer: &mut Timer,
    rng: &mut rand_chacha::ChaCha20Rng,
) -> i64 {
    const T0: f64 = 10.0;
    const T1: f64 = 0.1;
    let mut temp = T0;
    let mut prob;

    let mut count = 0;
    let mut now_score = compute_score(input, output);

    let mut best_score = now_score;
    let mut best_output = output.clone();
    loop {
        if count >= 100 {
            // if count % 100 == 0 {
            let passed = timer.get_time() / TIMELIMIT;
            if passed >= 1.0 {
                break;
            }
            // if count % 1000 == 0 {
            //     best_output.write();
            // }
            // eprintln!("{} {}", temp, now_score);
            temp = T0.powf(1.0 - passed) * T1.powf(passed);
            // temp = s_temp.powf(1.0 - passed) * e_temp.powf(passed);
            count = 0;

            // 頂点0につながっていない頂点のpowerを0に
            let is_connected = output.get_connection_status(input);

            for (i, b) in is_connected.into_iter().enumerate() {
                if !b {
                    output.powers[i] = 0;
                }
            }
        }
        count += 1;

        let mut new_out = output.clone();
        // 近傍解生成。powers と edges について同時焼きなまし
        // powers について
        let i = rng.gen_range(0, input.n);
        if new_out.powers[i] >= 4590 {
            new_out.powers[i] -= rng.gen_range(2, 25);
        } else if new_out.powers[i] <= 50 {
            new_out.powers[i] += rng.gen_range(2, 25);
        } else if rng.gen_bool(0.5) {
            new_out.powers[i] -= rng.gen_range(2, 25);
        } else {
            new_out.powers[i] += rng.gen_range(2, 25);
        }
        // edges について
        for _ in 0..5 {
            let i = rng.gen_range(0, input.m);
            new_out.edges[i] ^= true;
        }
        let new_score = compute_score(input, &new_out);
        prob = f64::exp((new_score - now_score) as f64 / temp);
        if now_score < new_score || rng.gen_bool(prob) {
            now_score = new_score;
            *output = new_out;
        }

        if best_score < now_score {
            best_score = now_score;
            best_output = output.clone();
        }
    }
    // eprintln!("{}", best_score);
    *output = best_output;
    best_score
}

fn compute_score(input: &Input, out: &Output) -> i64 {
    let broadcasted_count = out.get_broadcasted_count(input);

    if broadcasted_count < input.k {
        (1e6 * (broadcasted_count + 1) as f64 / input.k as f64).round() as i64
    } else {
        let cost = out.calc_cost(input);
        (1e6 * (1.0 + 1e8 / (cost as f64 + 1e7))).round() as i64
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    pub fn calc_sq_dist(&self, other: &Point) -> i32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

pub struct Input {
    pub n: usize,
    pub m: usize,
    pub k: usize,
    pub stations: Vec<Point>,
    pub edges: Vec<(usize, usize, i32)>,
    pub residents: Vec<Point>,
}

fn read_input() -> Input {
    use proconio::{input, marker::Usize1};
    input! {
        n: usize,
        m: usize,
        k: usize,
        stations: [(i32, i32); n],
        edges: [(Usize1, Usize1, i32); m],
        residents: [(i32, i32); k],
    }

    Input {
        n,
        m,
        k,
        stations: stations.iter().map(|&(x, y)| Point::new(x, y)).collect(),
        edges,
        residents: residents.iter().map(|&(x, y)| Point::new(x, y)).collect(),
    }
}

#[derive(Debug, Clone)]
pub struct Output {
    pub powers: Vec<i32>,
    pub edges: Vec<bool>,
}

#[allow(dead_code, clippy::needless_range_loop)]
impl Output {
    fn get_broadcasted_count(&self, input: &Input) -> usize {
        let is_connected = self.get_connection_status(input);
        self.get_broadcasted_status(input, &is_connected)
            .iter()
            .filter(|&&b| b)
            .count()
    }

    fn get_connection_status(&self, input: &Input) -> Vec<bool> {
        let mut dsu = Dsu::new(input.n);

        for (j, used) in self.edges.iter().enumerate() {
            if !used {
                continue;
            }

            let (u, v, _) = input.edges[j];
            dsu.merge(u, v);
        }

        (0..input.n).map(|i| dsu.same(0, i)).collect()
    }

    fn get_broadcasted_status(&self, input: &Input, is_connected: &[bool]) -> Vec<bool> {
        let mut broadcasted = vec![false; input.k];

        for i in 0..input.n {
            if !is_connected[i] {
                continue;
            }

            for k in 0..input.k {
                let dist_sq = input.stations[i].calc_sq_dist(&input.residents[k]);
                let power = self.powers[i];
                broadcasted[k] |= dist_sq <= power * power;
            }
        }

        broadcasted
    }

    fn calc_cost(&self, input: &Input) -> i64 {
        let mut cost = 0;

        for i in 0..input.n {
            cost += self.calc_power_cost(i);
        }

        for (j, used) in self.edges.iter().enumerate() {
            if !used {
                continue;
            }

            let (_, _, w) = input.edges[j];
            cost += w as i64;
        }

        cost
    }

    fn calc_power_cost(&self, v: usize) -> i64 {
        let p = self.powers[v] as i64;
        p * p
    }

    fn write(&self) {
        for (i, p) in self.powers.iter().enumerate() {
            if i == self.powers.len() - 1 {
                println!("{}", p);
            } else {
                print!("{} ", p);
            }
        }

        for (i, b) in self.edges.iter().enumerate() {
            let n = if *b { 1 } else { 0 };
            if i == self.edges.len() - 1 {
                println!("{}", n);
            } else {
                print!("{} ", n);
            }
        }
    }
}

pub struct Dsu {
    n: usize,
    parent_or_size: Vec<i32>,
}

impl Dsu {
    pub fn new(size: usize) -> Self {
        Self {
            n: size,
            parent_or_size: vec![-1; size],
        }
    }

    pub fn merge(&mut self, a: usize, b: usize) -> usize {
        assert!(a < self.n);
        assert!(b < self.n);
        let (mut x, mut y) = (self.leader(a), self.leader(b));
        if x == y {
            return x;
        }
        if -self.parent_or_size[x] < -self.parent_or_size[y] {
            std::mem::swap(&mut x, &mut y);
        }
        self.parent_or_size[x] += self.parent_or_size[y];
        self.parent_or_size[y] = x as i32;
        x
    }

    pub fn same(&mut self, a: usize, b: usize) -> bool {
        assert!(a < self.n);
        assert!(b < self.n);
        self.leader(a) == self.leader(b)
    }

    pub fn leader(&mut self, a: usize) -> usize {
        assert!(a < self.n);
        if self.parent_or_size[a] < 0 {
            return a;
        }
        self.parent_or_size[a] = self.leader(self.parent_or_size[a] as usize) as i32;
        self.parent_or_size[a] as usize
    }

    pub fn size(&mut self, a: usize) -> usize {
        assert!(a < self.n);
        let x = self.leader(a);
        -self.parent_or_size[x] as usize
    }

    pub fn groups(&mut self) -> Vec<Vec<usize>> {
        let mut leader_buf = vec![0; self.n];
        let mut group_size = vec![0; self.n];
        for i in 0..self.n {
            leader_buf[i] = self.leader(i);
            group_size[leader_buf[i]] += 1;
        }
        let mut result = vec![Vec::new(); self.n];
        for i in 0..self.n {
            result[i].reserve(group_size[i]);
        }
        for i in 0..self.n {
            result[leader_buf[i]].push(i);
        }
        result
            .into_iter()
            .filter(|x| !x.is_empty())
            .collect::<Vec<Vec<usize>>>()
    }
}

fn get_time() -> f64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9
}

struct Timer {
    start_time: f64,
}

impl Timer {
    fn new() -> Timer {
        Timer {
            start_time: get_time(),
        }
    }

    fn get_time(&self) -> f64 {
        get_time() - self.start_time
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.start_time = 0.0;
    }
}
