use std::collections::VecDeque;

use rand::prelude::*;
use rustc_hash::FxHashMap;

const TIMELIMIT: f64 = 1.65;

fn main() {
    let mut timer = Timer::new();
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let input = read_input();
    let mut output = greedy(&input);
    annealing(&input, &mut output, &mut timer, &mut rng);
    // 葉であって、出力が0の駅につながる辺を削除
    // 木を見ているので、葉であることは次数1と同じ
    let mut degree = vec![0; input.n];
    let mut g = vec![vec![]; input.n];
    let mut edge_to_index_map = FxHashMap::default();
    for (i, b) in output.edges.iter().enumerate() {
        if *b {
            let e = &input.edges[i];
            degree[e.0] += 1;
            degree[e.1] += 1;
            g[e.0].push(e.1);
            g[e.1].push(e.0);
            if e.0 < e.1 {
                edge_to_index_map.insert((e.0, e.1), i);
            } else {
                edge_to_index_map.insert((e.1, e.0), i);
            }
        }
    }
    let mut que = VecDeque::new();
    for (i, d) in degree.iter().enumerate() {
        if *d == 1 {
            que.push_back(i);
        }
    }
    while !que.is_empty() {
        let v = que.pop_front().unwrap();
        if v != 0 && output.powers[v] == 0 {
            for &u in g[v].iter() {
                let i = if u < v {
                    edge_to_index_map[&(u, v)]
                } else {
                    edge_to_index_map[&(v, u)]
                };
                if !output.edges[i] {
                    continue;
                }
                output.edges[i] = false;
                degree[u] -= 1;
                if degree[u] == 1 {
                    que.push_back(u);
                }
            }
        }
    }
    output.write();
    // eprintln!("{}", compute_score(&input, &output));
}

fn greedy(input: &Input) -> Output {
    let mut p = vec![0; input.n];
    // 客を見て、入ってないやつがいたら一番近い頂点のパワーを調整
    // let mut nearest_station_from_resident = vec![(0, 0); input.k];
    for (_, r) in input.residents.iter().enumerate() {
        let mut min_dist = (r.calc_sq_dist(&input.stations[0]) as f64).sqrt().ceil() as i32;
        let mut min_i = 0;
        let mut contain = false;
        for (i, station) in input.stations.iter().enumerate() {
            let dist = r.calc_sq_dist(station);
            let dist = (dist as f64).sqrt().ceil() as i32;
            if min_dist > dist {
                min_dist = dist;
                min_i = i;
            }
            if dist <= p[i] {
                contain = true;
            }
        }
        // nearest_station_from_resident[k] = (min_i, min_dist);
        if !contain {
            p[min_i] = min_dist;
        }
    }
    // 頂点0につながっている頂点で、現在の出力範囲内にいる最も遠い客は入るように出力を小さくする
    // 客が出力範囲内にいないときは、出力を0にする
    for (i, station) in input.stations.iter().enumerate() {
        let mut max_dist = 0;
        for r in input.residents.iter() {
            let dist = r.calc_sq_dist(station);
            let dist = (dist as f64).sqrt().ceil() as i32;
            if dist <= p[i] && max_dist < dist {
                max_dist = dist;
            }
        }
        p[i] = max_dist;
    }
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
    const T0: f64 = 100.0;
    const T1: f64 = 100.0;
    let mut temp = T0;
    let mut prob;

    let mut count = 0;
    let mut now_score = compute_score(input, output);

    let mut best_score = now_score;
    let mut best_output = output.clone();
    loop {
        if count >= 100 {
            let passed = timer.get_time() / TIMELIMIT;
            if passed >= 1.0 {
                break;
            }
            temp = T0.powf(1.0 - passed) * T1.powf(passed);
            count = 0;
        }
        count += 1;

        let mut new_out = output.clone();
        // 近傍解生成。powers について同時焼きなまし
        // powers について
        let rng_i = rng.gen_range(0, input.n);
        new_out.powers[rng_i] = 0;
        for (_, r) in input.residents.iter().enumerate() {
            let mut min_dist = (r.calc_sq_dist(&input.stations[0]) as f64).sqrt().ceil() as i32;
            let mut min_i = 0;
            let mut contain = false;
            for (i, station) in input.stations.iter().enumerate() {
                if rng_i == i {
                    continue;
                }
                let dist = r.calc_sq_dist(station);
                let dist = (dist as f64).sqrt().ceil() as i32;
                if min_dist > dist {
                    min_dist = dist;
                    min_i = i;
                }
                if dist <= new_out.powers[i] {
                    contain = true;
                }
            }
            if !contain {
                new_out.powers[min_i] = min_dist;
            }
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
