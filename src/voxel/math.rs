use std::{ops::{Mul, Add, Sub}, fmt::{Display, Formatter, Result}};

use bytemuck::{Pod, Zeroable};

#[derive(Debug, Copy, Clone)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

#[derive(Debug, Copy, Clone)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T
}

#[derive(Debug, Copy, Clone)]
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T
}

impl<T> Vec2<T> {

    pub fn new(x: T, y: T) -> Self {
        Vec2{x, y}
    }

    pub fn start(self) -> T {
        self.x
    }

    pub fn end(self) -> T {
        self.y
    }
}

impl<T: PartialOrd> Vec2<T> {
    /// Returns true if `self` is in range of `of_other_vec`
    pub fn in_range(self, of_other_vec: Self) -> bool {
        self.x >= of_other_vec.x && self.y <= of_other_vec.y
    }

    pub fn overlapping(self, other: Self) -> bool {
        (other.x <= self.x && other.y >= self.y)
        || (other.x >= self.x && other.y >= self.y && other.x <= self.y)
        || (other.x <= self.x && other.y <= self.y && other.y >= self.x)
        // (self.x <= other.x && self.y >= other.y)
        // || (self.x >= other.x && self.y >= other.y && self.x <= other.y)
        // || (self.x <= other.x && self.y <= other.y && self.y >= other.x)
    }
}

impl<T: Eq> Vec2<T> {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl<T: Sub<Output=T>> Vec2<T> {
    /// # NOTE!
    /// Returns may be negative! Be sure to convert to positive using .abs() if applicable
    pub fn length(self) -> T {
        self.x - self.y
    }
}

impl<T: Display> Display for Vec2<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl<T> Vec3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Vec3{x, y, z}
    }
}

impl<T: Display> Display for Vec3<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl<T> Vec4<T> {
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        Vec4{x, y, z, w}
    }

    pub fn r(self) -> T {
        self.x
    }

    pub fn g(self) -> T {
        self.y
    }

    pub fn b(self) -> T {
        self.z
    }

    pub fn a(self) -> T {
        self.w
    }
}

impl<T: Mul<Output=T>> Mul for Vec4<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z, self.w * rhs.w)
    }
}

impl<T: Add<Output=T>> Add for Vec4<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z, self.w + rhs.w)
    }
}

impl<T: Display> Display for Vec4<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}