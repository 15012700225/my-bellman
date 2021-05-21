//use fs2::FileExt;
use crate::sector_id::SECTOR_ID;
use log::{debug, info, warn};
//use std::fs::File;
//use std::path::PathBuf;

// const GPU_LOCK_NAME: &str = "bellman.gpu.lock";
// const PRIORITY_LOCK_NAME: &str = "bellman.priority.lock";
//fn tmp_path(filename: &str) -> PathBuf {
//    let mut p = std::env::temp_dir();
//    p.push(filename);
//    p
//}

/// `GPULock` prevents two kernel objects to be instantiated simultaneously.
#[derive(Debug)]
pub struct GPULock();
impl GPULock {
    pub fn lock() -> GPULock {
        debug!("{:?}: Acquiring GPU lock...", *SECTOR_ID);
        debug!("{:?}: GPU lock acquired!", *SECTOR_ID);
        GPULock()
    }
}
impl Drop for GPULock {
    fn drop(&mut self) {
        debug!("GPU lock released!");
    }
}

/// `PrioriyLock` is like a flag. When acquired, it means a high-priority process
/// needs to acquire the GPU really soon. Acquiring the `PriorityLock` is like
/// signaling all other processes to release their `GPULock`s.
/// Only one process can have the `PriorityLock` at a time.
#[derive(Debug)]
pub struct PriorityLock();
impl PriorityLock {
    pub fn lock() -> PriorityLock {
        debug!("{:?}: Acquiring priority lock...", *SECTOR_ID);
        debug!("Priority lock acquired!");
        PriorityLock()
    }
    pub fn wait(_priority: bool) {}
    pub fn should_break(_priority: bool) -> bool {
        false
    }
}
impl Drop for PriorityLock {
    fn drop(&mut self) {
        debug!("Priority lock released!");
    }
}

use super::error::{GPUError, GPUResult};
use super::fft::FFTKernel;
use super::multiexp::MultiexpKernel;
use crate::bls::Engine;
use crate::domain::create_fft_kernel;
use crate::multiexp::create_multiexp_kernel;

macro_rules! locked_kernel {
    ($class:ident, $kern:ident, $func:ident, $name:expr) => {
        pub struct $class<E>
        where
            E: Engine,
        {
            log_d: usize,
            priority: bool,
            kernel: Option<$kern<E>>,
            gpu_index: usize,
        }

        impl<E> $class<E>
        where
            E: Engine,
        {
            pub fn new(log_d: usize, priority: bool, gpu_index: usize) -> $class<E> {
                $class::<E> {
                    log_d,
                    priority,
                    kernel: None,
                    gpu_index,
                }
            }

            pub fn init(&mut self) {
                if self.kernel.is_none() {
                    PriorityLock::wait(self.priority);
                    info!("{:?}: GPU is available for {}!", *SECTOR_ID, $name);
                    self.kernel = $func::<E>(self.log_d, self.priority, self.gpu_index);
                }
            }

            fn free(&mut self) {
                if let Some(_kernel) = self.kernel.take() {
                    warn!(
                        "{:?}: GPU acquired by a high priority process! Freeing up {} kernels...",
                        *SECTOR_ID, $name
                    );
                }
            }

            pub fn with<F, R>(&mut self, mut f: F) -> GPUResult<R>
            where
                F: FnMut(&mut $kern<E>) -> GPUResult<R>,
            {
                if std::env::var("BELLMAN_NO_GPU").is_ok() {
                    return Err(GPUError::GPUDisabled);
                }

                self.init();

                loop {
                    if let Some(ref mut k) = self.kernel {
                        match f(k) {
                            Err(GPUError::GPUTaken) => {
                                warn!("{:?}: GPU taken, re-initializing", *SECTOR_ID);
                                self.free();
                                self.init();
                            }
                            Err(e) => {
                                warn!(
                                    "{:?}: GPU {} failed! Falling back to CPU... Error: {}",
                                    *SECTOR_ID, $name, e
                                );
                                return Err(e);
                            }
                            Ok(v) => return Ok(v),
                        }
                    } else {
                        return Err(GPUError::KernelUninitialized);
                    }
                }
            }
        }
    };
}

locked_kernel!(LockedFFTKernel, FFTKernel, create_fft_kernel, "FFT");
locked_kernel!(
    LockedMultiexpKernel,
    MultiexpKernel,
    create_multiexp_kernel,
    "Multiexp"
);
