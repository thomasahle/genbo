//! Memory-mapped reader for the big-ann `.fbin` float32 format:
//! 8-byte header (u32 little-endian `nb`, u32 `d`) followed by `nb * d` `f32` values.
//! Used for the OOD FLOAT-RERANK path: the index/scan stay int8, but the exact rerank of the
//! t_surv survivors reads the ORIGINAL float vectors (only survivors are touched), so ranking is
//! at float precision against the leaderboard's float-computed ground truth. mmap'd: rows page in
//! on first touch, so reading only survivors keeps the float base near-zero in RAM.

use memmap2::Mmap;
use std::fs::File;
use std::io;

pub struct FBin {
    _mmap: Mmap,
    pub nb: usize,
    pub d: usize,
    base: *const f32, // start of the vector region (after the 8-byte header)
}

// SAFETY: read-only immutable mmap for the lifetime of FBin; rows are disjoint reads.
unsafe impl Send for FBin {}
unsafe impl Sync for FBin {}

impl FBin {
    /// Open a `.fbin`, optionally restricting to the first `limit` rows (0 = all). The header dim
    /// is honored; the file region is asserted large enough. The data region (offset 8) is 4-byte
    /// aligned, so the f32 reinterpret is valid.
    pub fn open(path: &str, limit: usize) -> io::Result<Self> {
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        assert!(mmap.len() >= 8, "fbin too small for header");
        let nb_file = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let d = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        let nb = if limit == 0 { nb_file } else { limit.min(nb_file) };
        assert!(mmap.len() >= 8 + nb * d * 4, "fbin truncated: have {} need {}", mmap.len(), 8 + nb * d * 4);
        assert!((mmap.as_ptr() as usize + 8) % 4 == 0, "fbin data region not 4-byte aligned");
        let base = unsafe { mmap.as_ptr().add(8) } as *const f32;
        Ok(Self { _mmap: mmap, nb, d, base })
    }

    /// Borrow row `i` as a `&[f32]` of length `d` (pages in from disk on first touch).
    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        debug_assert!(i < self.nb);
        unsafe { std::slice::from_raw_parts(self.base.add(i * self.d), self.d) }
    }
}
