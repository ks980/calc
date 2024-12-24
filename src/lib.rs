use nalgebra;
use nalgebra::DMatrix;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_complex::{Complex64, ComplexFloat};
use numpy::{PyArray1, PyArray2};
use pyo3::{prelude::*, types::PyList};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::f64::consts::PI;

#[pyclass]
#[derive(Copy, Clone)]
pub struct Slit {
    center: f64,
    width: f64,
}

#[pymethods]
impl Slit {
    #[new]
    pub fn new(center: f64, width: f64) -> Self {
        Self { center, width }
    }

    pub fn get_center(&self) -> PyResult<f64> {
        Ok(self.center)
    }

    pub fn get_width(&self) -> f64 {
        self.width
    }
}

#[pyclass]
#[derive(Clone)]
pub struct GaussianBeam {
    density_matrix: Array2<Complex64>,
    coordination: Array1<Complex64>,
    discrete_step_size: f64,
    discrete_step_num: usize,
    wl: f64,
}

#[pymethods]
impl GaussianBeam {
    #[new]
    pub fn new(
        sigma: f64,
        distance: f64,
        center: f64,
        width: f64,
        discrete_step_size: f64,
        wl: f64,
    ) -> Self {
        let discrete_step_num = (width / discrete_step_size) as usize;
        let coordination =
            Array1::linspace(center - width / 2., center + width / 2., discrete_step_num)
                .map(|&x| Complex64::new(x, 0.));
        let psi1_vec = coordination.map(|&x| psi1(x, distance, sigma));
        let psi2_vec = coordination.map(|&x| psi2(x, distance, sigma));
        let density_matrix = self_mul_dagger(psi1_vec.view()) + self_mul_dagger(psi2_vec.view());
        Self {
            density_matrix,
            coordination,
            discrete_step_size,
            discrete_step_num,
            wl,
        }
    }

    pub fn get_density_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<Complex64>> {
        PyArray2::from_array(py, &self.density_matrix)
    }

    pub fn get_coordination<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex64>> {
        PyArray1::from_array(py, &self.coordination)
    }

    pub fn get_discrete_step_size(&self) -> f64 {
        self.discrete_step_size
    }

    pub fn get_discrete_step_num(&self) -> usize {
        self.discrete_step_num
    }

    pub fn get_wl(&self) -> f64 {
        self.wl
    }

    pub fn pass_slit_in_place(&mut self, slit: &Slit) {
        let slit1_p = construct_slit_p(slit.center, slit.width, self.coordination.view());
        self.density_matrix = slit1_p
            .dot(&self.density_matrix)
            .dot(&slit1_p.t().map(|x| x.conj()));
        //self.density_matrix /= self.density_matrix.diag().sum();
    }
    pub fn pass_len_in_place(&mut self, focal_len: f64) {
        // let mut out: Array2<Complex64> =
        //     Array2::zeros((self.discrete_step_num, self.discrete_step_num));
        // let mut intermedia = out.clone();
        //
        // let fft_handler = FftHandler::new(self.discrete_step_num);
        // ndfft(&self.density_matrix, &mut intermedia, &fft_handler, 0);
        // ndfft(&intermedia, &mut out, &fft_handler, 1);
        let w = dft_matrix(self.discrete_step_num);
        let out = w
            .clone()
            .dot(&self.density_matrix)
            .dot(&w.t().map(|x| x.conj()));
        //let out_tr = out.diag().sum();
        self.density_matrix = out; // out_tr;

        let new_discrete_step_size =
            self.wl * focal_len / self.discrete_step_size / self.discrete_step_num as f64;

        self.coordination *= Complex64::new(new_discrete_step_size / self.discrete_step_size, 0.);
        self.discrete_step_size = new_discrete_step_size;
        fft_shift(&mut self.density_matrix);
    }
}
impl GaussianBeam {
    fn pass_slit(&mut self, slit: &Slit) -> &mut Self {
        let slit1_p = construct_slit_p(slit.center, slit.width, self.coordination.view());
        self.density_matrix = slit1_p
            .dot(&self.density_matrix)
            .dot(&slit1_p.t().map(|x| x.conj()));
        //self.density_matrix /= self.density_matrix.diag().sum();
        self
    }
    fn pass_len(&mut self, focal_len: f64) -> &mut Self {
        let mut out: Array2<Complex64> =
            Array2::zeros((self.discrete_step_num, self.discrete_step_num));
        //let mut intermedia = out.clone();
        // let fft_handler = FftHandler::new(self.discrete_step_num);
        // ndfft(&self.density_matrix, &mut intermedia, &fft_handler, 0);
        // ndfft(&intermedia, &mut out, &fft_handler, 1);
        let w = dft_matrix(self.discrete_step_num);
        let out = w
            .clone()
            .dot(&self.density_matrix)
            .dot(&w.t().map(|x| x.conj()));
        let out_tr = out.diag().sum();
        self.density_matrix = out; // out_tr;

        let new_discrete_step_size =
            self.wl * focal_len / self.discrete_step_size / self.discrete_step_num as f64;

        self.coordination *= Complex64::new(new_discrete_step_size / self.discrete_step_size, 0.);
        self.discrete_step_size = new_discrete_step_size;
        fft_shift(&mut self.density_matrix);
        self
    }
}
fn psi1(x: Complex64, d: f64, sigma: f64) -> Complex64 {
    1. / (2. * PI * sigma * sigma).sqrt() * (-0.5 * ((x - 0.5 * d) / sigma).powi(2)).exp()
}

fn psi2(x: Complex64, d: f64, sigma: f64) -> Complex64 {
    psi1(x, -d, sigma)
}

fn construct_slit_p(center: f64, width: f64, cord: ArrayView1<Complex64>) -> Array2<Complex64> {
    let start = center - width * 0.5;
    let end = center + width * 0.5;
    Array2::from_diag(&cord.map(|&c| c.re > start && c.re < end).map(|v: &bool| {
        if *v {
            Complex64::new(1., 0.)
        } else {
            Complex64::new(0., 0.)
        }
    }))
}

fn self_mul_dagger(v: ArrayView1<Complex64>) -> Array2<Complex64> {
    v.insert_axis(Axis(1))
        .dot(&v.insert_axis(Axis(0)).map(|&x| x.conj()))
}
//https://en.wikipedia.org/wiki/DFT_matrix
fn dft_matrix(n: usize) -> Array2<Complex64> {
    let omega = Complex64::new(0., -2. * PI / n as f64).exp();
    let w = Array2::from_shape_fn((n, n), |(i, j)| omega.powi((i * j) as i32)) / (n as f64).sqrt();
    w
}

fn fft_shift(array: &mut Array2<Complex64>) {
    let (rows, cols) = (array.nrows(), array.ncols());
    let (row_mid, col_mid) = (rows / 2, cols / 2);
    let temp_array = array.clone();
    // Swap quadrants
    array
        .slice_mut(s![..row_mid, ..col_mid])
        .assign(&temp_array.slice(s![row_mid.., col_mid..]));
    array
        .slice_mut(s![row_mid.., col_mid..])
        .assign(&temp_array.slice(s![..row_mid, ..col_mid]));
    array
        .slice_mut(s![..row_mid, col_mid..])
        .assign(&temp_array.slice(s![row_mid.., ..col_mid]));
    array
        .slice_mut(s![row_mid.., ..col_mid])
        .assign(&temp_array.slice(s![..row_mid, col_mid..]));
}

#[pyfunction]
pub fn center_mapping<'py>(
    light: &GaussianBeam,
    slit1_array: Vec<Slit>,
    slit2: Slit,
    focal_len: f64,
    py: Python<'py>,
) -> Bound<'py, MapResult> {
    let res: Vec<_> = slit1_array
        .par_iter()
        .map(|x| {
            let mut beam = light.clone();
            beam.pass_slit(x).pass_len(focal_len).pass_slit(&slit2);
            beam.density_matrix
        })
        // .map(|x| PyArray2::from_array(py, &x))
        .collect();
    let res: Vec<_> = res
        .into_iter()
        //.map(|x| PyArray2::from_array(py, &x))
        .collect();
    //PyList::new(py, res).unwrap()
    Bound::new(py, MapResult::new(res, light.discrete_step_size)).unwrap()
}
#[pyclass]
pub struct MapResult {
    density_matrix: Vec<Array2<Complex64>>,
    dx: f64,
}
impl MapResult {
    fn new(density_matrix: Vec<Array2<Complex64>>, dx: f64) -> Self {
        Self { density_matrix, dx }
    }
    fn qfi(&self,dx:f64) -> Array2<Complex64> {
        qfi(&self.density_matrix, dx)
    }
}
#[pymethods]
impl MapResult {
    pub fn calc_qfi<'py>(&self, py: Python<'py>, dx:f64) -> Bound<'py, PyArray2<Complex64>> {
        PyArray2::from_array(py, &self.qfi(dx))
    }
    pub fn get_density_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let res = self
            .density_matrix
            .iter()
            .map(|x| PyArray2::from_array(py, x))
            .collect::<Vec<_>>();
        PyList::new(py, res).unwrap()
    }
    pub fn get_dx<'py>(&self, _py: Python<'py>) -> f64 {
        self.dx
    }
}

fn eigen(arr: ArrayView2<Complex64>) -> (Array1<f64>, Array2<Complex64>) {
    let (rows, cols) = arr.dim();
    let data = arr.as_slice().expect("Memory not contiguous");
    let n_arr = DMatrix::from_row_slice(rows, cols, data);
    let x = n_arr.symmetric_eigen();
    let eigenvalues = Array1::from_vec(x.eigenvalues.as_slice().to_vec());
    let eigenvectors = Array2::from_shape_vec((rows, cols), x.eigenvectors.as_slice().to_vec())
        .expect("Failed to convert eigenvectors to Array2");
    // let arr_check =
    //     Array2::from_shape_vec((rows, cols), x.recompose().as_slice().to_vec()).unwrap();
    // let err = (arr_check.clone() - arr).sum().abs();
    // if err > 1e-9 {
    //     panic!(
    //         "wrong eigen {}\n{}",
    //         arr_check.clone().to_string(),
    //         arr.to_string()
    //     );
    // }
    (eigenvalues.to_owned(), eigenvectors)
}

fn qfi(rhos: &Vec<Array2<Complex64>>, dx: f64) -> Array2<Complex64> {
    let rhos_len = rhos.len();
    let out: Vec<Vec<Complex64>> = rhos
        .par_iter()
        .map(|x| {
            let mut full_rhos = VecDeque::new();
            let mut res = Vec::new();
            for i in 0..rhos_len {
                full_rhos.push_front(
                    x.clone() + rhos[i].view() + x.dot(&rhos[i]) + rhos[i].view().dot(x),
                );
                let mut sum = Complex64::new(0.0, 0.0);
                if i > 1 {
                    full_rhos.truncate(3);
                    let drhodx =
                        (full_rhos.front().unwrap() - full_rhos.back().unwrap()) / dx.powi(2);
                    let (eigv, eigm) = eigen(full_rhos.get(1).unwrap().view());
                    for ie in 0..eigv.len() {
                        for je in 0..eigv.len() {
                            let dominator = eigv[ie] + eigv[je];
                            if dominator.abs() < 1e-12 {
                                continue;
                            }
                            let sum1 = eigm
                                .slice(s![.., ie])
                                .map(|x| x.conj())
                                .dot(&drhodx)
                                .dot(&eigm.slice(s![.., je]).t())
                                .abs()
                                .powi(2)
                                / dominator;
                            sum += sum1;
                            // if sum.is_nan() {
                            //     panic!(
                            //         "sum is nan at {},{}, eigv={},{}, sum1 = {}\n eigm1={}\neigm2={}\n drhodx={}",
                            //         ie,
                            //         je,
                            //         eigv[ie],
                            //         eigv[je],
                            //         sum1,
                            //         eigm.slice(s![.., ie]).to_string(),
                            //         eigm.slice(s![.., je]).to_string(),
                            //         drhodx.to_string()
                            //     );
                            //}
                            // sum += sum1;
                        }
                    }
                }
                res.push(sum);
            }
            res
        })
        .collect();
    out.iter().for_each(|x| {
        x.iter().for_each(|x| {
            if x.re.is_nan() || x.im.is_nan() {
                panic!("Nan")
            }
        })
    });
    Array2::from_shape_fn((rhos_len, rhos_len), |(i, j)| out[i][j])
}

#[pymodule]
fn gbeam(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Slit>()?;
    m.add_class::<GaussianBeam>()?;
    m.add_function(wrap_pyfunction!(center_mapping, m)?)?;
    Ok(())
}
