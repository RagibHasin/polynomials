#![no_std]
use core::cmp::PartialEq;
use core::convert::From;
use core::ops::Neg;
use core::ops::{Add, AddAssign};
use core::ops::{Div, DivAssign};
use core::ops::{Index, IndexMut};
use core::ops::{Mul, MulAssign};
use core::ops::{Sub, SubAssign};
use core::slice::SliceIndex;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(test, macro_use)]
extern crate alloc;
use alloc::{
    vec,
    vec::{IntoIter, Vec},
};

/// A [`Polynomial`] is just a vector of coefficients. Each coefficient corresponds to a power of
/// `x` in increasing order. For example, the following polynomial is equal to 4x^2 + 3x - 9.
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// // Construct polynomial 4x^2 + 3x - 9
/// let mut a = poly![-9, 3, 4];
/// assert_eq!(a[0], -9);
/// assert_eq!(a[1], 3);
/// assert_eq!(a[2], 4);
/// # }
/// ```
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Polynomial<T>(Vec<T>);

impl<T> Polynomial<T> {
    /// Create a new, empty, instance of a polynomial.
    pub fn new() -> Polynomial<T> {
        Polynomial(Vec::<T>::new())
    }

    /// Adds a new coefficient to the [`Polynomial`], in the next highest order position.
    ///
    /// ```
    /// # #[macro_use] extern crate polynomials;
    /// # fn main() {
    /// let mut a = poly![-8, 2, 4];
    /// a.push(7);
    /// assert_eq!(a, poly![-8, 2, 4, 7]);
    /// # }
    /// ```
    pub fn push(&mut self, value: T) {
        self.0.push(value);
    }

    /// Removes the highest order coefficient from the [`Polynomial`].
    ///
    /// ```
    /// # #[macro_use] extern crate polynomials;
    /// # fn main() {
    /// let mut a = poly![-8, 2, 4];
    /// assert_eq!(a.pop().unwrap(), 4);
    /// assert_eq!(a, poly![-8, 2]);
    /// # }
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

    /// Calculates the degree of a [`Polynomial`].
    ///
    /// The following polynomial is of degree 2: (4x^2 + 2x - 8)
    ///
    /// ```
    /// # #[macro_use] extern crate polynomials;
    /// # fn main() {
    /// let a = poly![-8, 2, 4];
    /// assert_eq!(a.degree(), 2);
    /// # }
    /// ```
    pub fn degree(&self) -> usize
    where
        T: Sub<T, Output = T> + PartialEq + Copy + Default,
    {
        let mut deg = self.0.len();
        for _ in 0..self.0.len() {
            deg -= 1;

            // Generic test if non-zero
            if self[deg] != T::default() {
                break;
            }
        }
        deg
    }

    /// Evaluate a [`Polynomial`] for some value `x`.
    ///
    /// The following example evaluates the polynomial (4x^2 + 2x - 8) for x = 3.
    ///
    /// ```
    /// # #[macro_use] extern crate polynomials;
    /// # fn main() {
    /// let a = poly![-8, 2, 4];
    /// assert_eq!(a.eval(3), 34);
    /// # }
    /// ```
    pub fn eval<X>(&self, x: X) -> T
    where
        T: AddAssign + Copy + Default,
        X: MulAssign + Mul<T, Output = T> + Copy,
    {
        if self.0.is_empty() {
            T::default()
        } else {
            let mut p = x; // running power of `x`
            let mut res = self[0];
            for i in 1..self.0.len() {
                res += p * self[i];
                p *= x;
            }
            res
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }
}

impl<T> Default for Polynomial<T>
where
    T: Default,
{
    fn default() -> Self {
        poly![T::default()]
    }
}

impl<T> From<Vec<T>> for Polynomial<T> {
    fn from(v: Vec<T>) -> Self {
        Polynomial(v)
    }
}

impl<T> Into<Vec<T>> for Polynomial<T> {
    fn into(self) -> Vec<T> {
        self.0
    }
}

impl<T> IntoIterator for Polynomial<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        self.0.into_iter()
    }
}

impl<T, I: SliceIndex<[T]>> Index<I> for Polynomial<T> {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, I: SliceIndex<[T]>> IndexMut<I> for Polynomial<T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// Add two [`Polynomial`]s.
///
/// The following example adds two polynomials:
/// (4x^2 + 2x - 8) + (x + 1) = (4x^2 + 3x - 7)
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let a = poly![-8, 2, 4];
/// let b = poly![1, 1];
/// assert_eq!(a + b, poly![-7, 3, 4]);
/// # }
/// ```
impl<T: Add<Output = T>> Add for Polynomial<T>
where
    T: Add + Copy + Clone,
{
    type Output = Self;

    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

impl<T> AddAssign for Polynomial<T>
where
    T: Add<Output = T> + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        let min_len = if self.0.len() < rhs.0.len() {
            self.0.len()
        } else {
            rhs.0.len()
        };
        if self.0.len() < min_len {
            for i in self.0.len()..min_len {
                self.push(rhs[i]);
            }
        }
        for i in 0..min_len {
            self[i] = self[i] + rhs[i];
        }
    }
}

/// Subtract two [`Polynomial`]s.
///
/// The following example subtracts two polynomials:
/// (4x^2 + 2x - 8) - (x + 1) = (4x^2 + x - 9)
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let a = poly![-8, 2, 4];
/// let b = poly![1, 1];
/// assert_eq!(a - b, poly![-9, 1, 4]);
/// # }
/// ```
impl<T: Sub<Output = T>> Sub for Polynomial<T>
where
    T: Sub + Neg<Output = T> + Copy + Clone,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let mut diff = self;
        diff -= other;
        diff
    }
}

impl<T> SubAssign for Polynomial<T>
where
    T: Sub<Output = T> + Neg<Output = T> + Copy,
{
    fn sub_assign(&mut self, rhs: Self) {
        let min_len = if self.0.len() < rhs.0.len() {
            self.0.len()
        } else {
            rhs.0.len()
        };
        if self.0.len() < min_len {
            for i in self.0.len()..min_len {
                self.push(-rhs[i]);
            }
        }
        for i in 0..min_len {
            self[i] = self[i] - rhs[i];
        }
    }
}

/// Multiply two [`Polynomial`]s.
///
/// The following example multiplies two polynomials:
/// (4x^2 + 2x - 8) * (x + 1) = (4x^3 + 6x^2 - 6x - 8)
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let a = poly![-8, 2, 4];
/// let b = poly![1, 1];
/// assert_eq!(a * b, poly![-8, -6, 6, 4]);
/// # }
/// ```
impl<T> Mul<T> for Polynomial<T>
where
    T: MulAssign + Copy,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut prod = self;
        prod *= rhs;
        prod
    }
}

impl<T> MulAssign<T> for Polynomial<T>
where
    T: MulAssign + Copy,
{
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..self.0.len() {
            self[i] *= rhs;
        }
    }
}

/// Multiply a [`Polynomial`] by some value.
///
/// The following example multiplies a polynomial (4x^2 + 2x - 8) by 2:
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let p = poly![-8, 2, 4] * 2;
/// assert_eq!(p, poly![-16, 4, 8]);
/// # }
/// ```
impl<T> Mul for Polynomial<T>
where
    T: Mul<Output = T> + AddAssign + Sub<Output = T>,
    T: Copy + Clone,
    T: Default,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut new = self;
        new *= rhs;
        new
    }
}

impl<T> MulAssign for Polynomial<T>
where
    T: Mul<Output = T> + AddAssign + Sub<Output = T>,
    T: Copy + Clone,
    T: Default,
{
    fn mul_assign(&mut self, rhs: Self) {
        let orig = self.clone();

        // One of the vectors must be non-empty
        if !self.0.is_empty() || !rhs.0.is_empty() {
            // Since core::num does not provide the `Zero()` trait
            // this hack lets us calculate zero from any generic
            let zero = T::default();

            // Clear `self`
            for i in 0..self.0.len() {
                self.0[i] = zero;
            }

            // Resize vector with size M + N - 1
            self.0.resize(self.0.len() + rhs.0.len() - 1, zero);

            // Calculate product
            for i in 0..orig.0.len() {
                for j in 0..rhs.0.len() {
                    self[i + j] += orig[i] * rhs[j];
                }
            }
        }
    }
}

/// Divide a [`Polynomial`] by some value.
///
/// The following example divides a polynomial (4x^2 + 2x - 8) by 2:
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let p = poly![-8, 2, 4] / 2;
/// assert_eq!(p, poly![-4, 1, 2]);
/// # }
/// ```
impl<T> Div<T> for Polynomial<T>
where
    T: DivAssign + Copy,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let mut prod = self;
        prod /= rhs;
        prod
    }
}

impl<T> DivAssign<T> for Polynomial<T>
where
    T: DivAssign + Copy,
{
    fn div_assign(&mut self, rhs: T) {
        for i in 0..self.0.len() {
            self[i] /= rhs;
        }
    }
}

impl<T> PartialEq for Polynomial<T>
where
    T: Sub<T, Output = T> + PartialEq + Copy + Default,
{
    fn eq(&self, other: &Self) -> bool {
        let degree = self.degree();
        if degree != other.degree() {
            return false;
        }
        for i in 0..degree {
            if self[i] != other[i] {
                return false;
            }
        }
        true
    }
}
impl<T> Eq for Polynomial<T> where T: Sub<T, Output = T> + Eq + Copy + Default {}

/// Creates a [`Polynomial`] from a list of coefficients in ascending order.
///
/// This is a wrapper around the `vec!` macro, to instantiate a polynomial from
/// a vector of coefficients.
///
/// `poly!` allows `Polynomial`s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a [`Polynomial`] containing a given list of coefficients:
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let p = poly![1, 2, 3]; // 3x^2 + 2x + 1
/// assert_eq!(p[0], 1);
/// assert_eq!(p[1], 2);
/// assert_eq!(p[2], 3);
/// # }
/// ```
///
/// - Create a [`Polynomial`] from a given coefficient and size:
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let p = poly![1; 3]; // x^2 + x + 1
/// assert_eq!(p, poly![1, 1, 1]);
/// # }
/// ```
#[macro_export]
macro_rules! poly {
    ($($args:tt)*) => (
         $crate::Polynomial::from(vec![$($args)*])
     );
}

#[cfg(test)]
mod tests {
    #[test]
    fn degree() {
        assert_eq!(poly![8, 6, 2, 3].degree(), 3);
        assert_eq!(poly![8, 6, 2, 3].degree(), 3);
        assert_eq!(poly![0, 0, 6, 2, 3].degree(), 4);
        assert_eq!(poly![0, 0].degree(), 0);
        assert_eq!(poly![0, 99].degree(), 1);
        assert_eq!(poly![99, 0].degree(), 0);
    }

    #[test]
    fn eval() {
        assert_eq!(poly![1, 1, 1, 1].eval(1), 4);
        assert_eq!(poly![-2, -2, -2, -2].eval(1), -8);
        assert_eq!(poly![100, 0, 0, 0].eval(9), 100);
        assert_eq!(poly![0, 1, 0, 0].eval(9), 9);
        assert_eq!(poly![0, 0, -1, 0].eval(9), -81);
        assert_eq!(poly![0, -9, 0, 40].eval(2), 302);
    }

    #[test]
    fn iter() {
        assert_eq!(poly![0, -9, 0, 40].iter().sum::<isize>(), 31);
    }

    #[test]
    fn add() {
        let a = poly![-200, 6, 2, 3, 53, 0, 0]; // Higher order 0s should be ignored
        let b = poly![-1, -6, -7, 0, 1000];
        let c = poly![-201, 0, -5, 3, 1053];
        assert_eq!(a.clone() + b.clone(), c);
        assert_eq!(b + a, c);
    }

    #[test]
    fn add_assign() {
        let mut a = poly![-200, 6, 2, 3, 53, 0, 0]; // Higher order 0s should be ignored
        let b = poly![-1, -6, -7, 0, 1000];
        let c = poly![-201, 0, -5, 3, 1053];
        a += b;
        assert_eq!(a, c);
    }

    #[test]
    fn sub() {
        let a = poly![-200, 6, 2, 3, 53, 0, 0]; // Higher order 0s should be ignored
        let b = poly![-1, -6, -7, 0, 1000];
        let c = poly![-199, 12, 9, 3, -947];
        let d = poly![199, -12, -9, -3, 947];
        assert_eq!(a.clone() - b.clone(), c);
        assert_eq!(b - a, d);
    }

    #[test]
    fn sub_assign() {
        let mut a = poly![-200, 6, 2, 3, 53, 0, 0]; // Higher order 0s should be ignored
        let b = poly![-1, -6, -7, 0, 1000];
        let c = poly![-199, 12, 9, 3, -947];
        a -= b;
        assert_eq!(a, c);
    }

    #[test]
    fn mul() {
        let a = poly![1, 0, 0]; // Higher order 0s should be ignored
        let b = poly![0];
        let c = poly![0];
        assert_eq!(a * b, c);

        let a = poly![-7];
        let b = poly![4];
        let c = poly![-28];
        assert_eq!(a * b, c);

        let a = poly![0, 1];
        let b = poly![4];
        let c = poly![0, 4];
        assert_eq!(a * b, c);

        let a = poly![0, -1];
        let b = poly![0, 1];
        let c = poly![0, 0, -1];
        assert_eq!(a * b, c);
    }

    #[test]
    fn mul_assign() {
        let mut a = poly![1, 0, 0]; // Higher order 0s should be ignored
        let b = poly![0];
        let c = poly![0];
        a *= b;
        assert_eq!(a, c);

        let mut a = poly![-7];
        let b = poly![4];
        let c = poly![-28];
        a *= b;
        assert_eq!(a, c);

        let mut a = poly![0, 1];
        let b = poly![4];
        let c = poly![0, 4];
        a *= b;
        assert_eq!(a, c);

        let mut a = poly![0, -1];
        let b = poly![0, 1];
        let c = poly![0, 0, -1];
        a *= b;
        assert_eq!(a, c);
    }

    #[test]
    fn mul_by_value() {
        let a = poly![1, 2, 3];
        let b = poly![2, 4, 6];
        assert_eq!(a * 2, b);

        let mut a = poly![1, 2, 3];
        let b = poly![2, 4, 6];
        a *= 2;
        assert_eq!(a, b);
    }

    #[test]
    fn div_by_value() {
        let a = poly![2, 4, 6];
        let b = poly![1, 2, 3];
        assert_eq!(a / 2, b);

        let mut a = poly![2, 4, 6];
        let b = poly![1, 2, 3];
        a /= 2;
        assert_eq!(a, b);
    }
}
