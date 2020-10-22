mod error;
use std::sync::Mutex;
pub use self::error::*;

#[cfg(feature = "gpu")]
mod locks;
#[cfg(feature = "gpu")]
pub use self::locks::*;

#[cfg(feature = "gpu")]
mod sources;
#[cfg(feature = "gpu")]
pub use self::sources::*;

#[cfg(feature = "gpu")]
mod utils;
#[cfg(feature = "gpu")]
pub use self::utils::*;

#[cfg(feature = "gpu")]
mod structs;
#[cfg(feature = "gpu")]
pub use self::structs::*;

#[cfg(feature = "gpu")]
mod fft;
#[cfg(feature = "gpu")]
pub use self::fft::*;

#[cfg(feature = "gpu")]
mod multiexp;
#[cfg(feature = "gpu")]
pub use self::multiexp::*;


#[derive(Debug)]
pub struct Queue<T> {
    qdata: Vec<T>,
}

impl <T> Queue<T> {
    fn new() -> Self {
        Queue{ qdata: Vec::new() }
    }

    fn push(&mut self, item: T) {
        self.qdata.push(item);
    }

    fn pop(&mut self) ->Option<T> {
        let l = self.qdata.len();

        if l > 0 {
            let v = self.qdata.remove(0);
            Some(v)
        } else {
            None
        }
    }

    fn len(&mut self) -> usize{
        self.qdata.len()
    }
}

#[cfg(not(feature = "gpu"))]
mod nogpu;
#[cfg(not(feature = "gpu"))]
pub use self::nogpu::*;

#[cfg(feature = "gpu")]
use ocl::Device;
#[cfg(feature = "gpu")]
lazy_static::lazy_static! {
    //pub static ref GPU_NVIDIA_DEVICES: Vec<Device> = get_devices(GPU_NVIDIA_PLATFORM_NAME).unwrap_or_default();
    pub static ref GPU_NVIDIA_DEVICES: Vec<Device> = get_devices(GPU_NVIDIA_PLATFORM_NAME).unwrap_or_default();
    //pub static ref GPU_NVIDIA_DEVICES_QUEUE:  Mutex<Queue<GpuDeviceInfo>> = Mutex::new(Queue::new());
}
