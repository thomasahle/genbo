//! Memory-mapped reader for the big-ann `.i8bin` format:
//! 8-byte header (u32 little-endian `nb`, u32 `d`) followed by `nb * d` `i8` values.
//! The base file is mmap'd, never materialized — at 1B this is ~100 GB on disk, ~0 in RAM
//! until rows are touched (the OS pages them in on demand).

use memmap2::Mmap;
use std::fs::File;
use std::io;

pub struct I8Bin {
    _mmap: Mmap, // kept alive; `data` borrows from it
    pub nb: usize,
    pub d: usize,
    base: *const i8, // start of the vector region (after the 8-byte header)
}

// SAFETY: the mmap is read-only and immutable for the lifetime of I8Bin; rows are disjoint reads.
unsafe impl Send for I8Bin {}
unsafe impl Sync for I8Bin {}

impl I8Bin {
    pub fn open(path: &str) -> io::Result<Self> {
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        assert!(mmap.len() >= 8, "file too small for header");
        let nb = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let d = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        assert!(
            mmap.len() >= 8 + nb * d,
            "file truncated: have {} need {}",
            mmap.len(),
            8 + nb * d
        );
        let base = unsafe { mmap.as_ptr().add(8) } as *const i8;
        Ok(Self { _mmap: mmap, nb, d, base })
    }

    /// Open a CONTIGUOUS sub-range of an `.i8bin` as a logical dataset of `count` rows:
    /// `row(i)` returns the original row `start + i`. Lets the `stream` workload build on rows
    /// [0,n_init) and insert rows [n_init,N) from ONE base file with no crop files / extra RAM.
    pub fn open_range(path: &str, start: usize, count: usize) -> io::Result<Self> {
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        assert!(mmap.len() >= 8, "file too small for header");
        let d = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        let nb_file = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        assert!(start + count <= nb_file, "range {}+{} exceeds file nb {}", start, count, nb_file);
        assert!(mmap.len() >= 8 + (start + count) * d, "file truncated for range");
        let base = unsafe { mmap.as_ptr().add(8 + start * d) } as *const i8;
        Ok(Self { _mmap: mmap, nb: count, d, base })
    }

    /// Borrow row `i` as a `&[i8]` of length `d` (pages in from disk on first touch).
    #[inline]
    pub fn row(&self, i: usize) -> &[i8] {
        debug_assert!(i < self.nb);
        unsafe { std::slice::from_raw_parts(self.base.add(i * self.d), self.d) }
    }
}

/// Memory-mapped reader for the big-ann `.fbin` format (same 8-byte u32 nb/d header, then nb*d f32).
/// Used by the streaming eval for an exact FLOAT rerank of the int8-generated candidates: the int8
/// index caps recall@10 at the quantization ceiling vs the official (float) GT, so the survivors are
/// re-scored against the original float vectors to lift recall toward 1.0 within the 1-hour budget.
pub struct FBin {
    _mmap: Mmap,
    pub nb: usize,
    pub d: usize,
    base: *const f32,
}

unsafe impl Send for FBin {}
unsafe impl Sync for FBin {}

impl FBin {
    pub fn open(path: &str) -> io::Result<Self> {
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        assert!(mmap.len() >= 8, "file too small for header");
        let nb = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let d = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        assert!(mmap.len() >= 8 + nb * d * 4, "fbin truncated: have {} need {}", mmap.len(), 8 + nb * d * 4);
        let base = unsafe { mmap.as_ptr().add(8) } as *const f32;
        Ok(Self { _mmap: mmap, nb, d, base })
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        debug_assert!(i < self.nb);
        unsafe { std::slice::from_raw_parts(self.base.add(i * self.d), self.d) }
    }
}
