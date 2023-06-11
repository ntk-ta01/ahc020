fn main() {
    let input = read_input();
    let out = greedy(&input);
    out.write();
}

fn greedy(input: &Input) -> Output {
    let p = vec![5000; input.n];
    let b = vec![true; input.m];

    Output {
        powers: p,
        edges: b,
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
