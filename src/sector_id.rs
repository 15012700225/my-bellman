use lazy_static::lazy_static;
#[derive(Copy, Clone, Debug)]
pub struct SectorId(u64);

lazy_static! {
    pub static ref SECTOR_ID: SectorId = SectorId(
        std::env::args()
            .nth(4)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    );
}
