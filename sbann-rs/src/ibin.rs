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
    // RESIDENT-I8 (P346, SBANN_RESIDENT_I8): owned anonymous copy of the vector region. File-backed
    // mmaps get 4KB pages only (no THP on this kernel/fs), so a scattered rescore pays a TLB miss +
    // page walk per row; with THP=[always] an anonymous copy is 2MB-paged (~500x fewer TLB entries).
    // When set, `base` points into this buffer instead of the mmap.
    _resident: Option<Vec<i8>>,
}

// SAFETY: the mmap is read-only and immutable for the lifetime of I8Bin; rows are disjoint reads.
unsafe impl Send for I8Bin {}
unsafe impl Sync for I8Bin {}

impl I8Bin {
    pub fn open(path: &str) -> io::Result<Self> {
        Self::open_with_data_offset(path, 8)
    }

    /// Open an i8bin-compatible file whose two-u32 header is followed by explicit padding.
    /// The graph-layout experiment uses offset 64 so row 0 starts on a cache-line boundary while
    /// retaining the ordinary (nb,d) header and dense d-byte row stride.
    pub fn open_with_data_offset(path: &str, data_offset: usize) -> io::Result<Self> {
        assert!(data_offset >= 8, "i8bin data offset must retain the 8-byte header");
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        assert!(mmap.len() >= 8, "file too small for header");
        let nb = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let d = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        assert!(
            mmap.len() >= data_offset + nb * d,
            "file truncated: have {} need {}",
            mmap.len(),
            data_offset + nb * d
        );
        let base = unsafe { mmap.as_ptr().add(data_offset) } as *const i8;
        Ok(Self { _mmap: mmap, nb, d, base, _resident: None })
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
        Ok(Self { _mmap: mmap, nb: count, d, base, _resident: None })
    }

    /// RESIDENT-I8 (P346): copy the vector region into anonymous memory (THP-eligible under
    /// `transparent_hugepage=[always]`) and repoint `base`. Kills the per-row TLB miss + page walk of
    /// scattered rescore gathers over the 4KB-paged file mmap. The live region is explicitly
    /// cache-line-aligned; large allocations are often page-aligned already, but Vec does not promise
    /// that and the graph-layout path relies on a stable 64-byte row origin. Byte-identical data.
    pub fn make_resident(&mut self) {
        let n = self.nb * self.d;
        let src = self.base;
        let mut buf = vec![0i8; n + 63];
        let misalignment = buf.as_ptr() as usize & 63;
        let offset = (64 - misalignment) & 63;
        buf[offset..offset + n]
            .copy_from_slice(unsafe { std::slice::from_raw_parts(src, n) });
        self.base = unsafe { buf.as_ptr().add(offset) };
        debug_assert_eq!(self.base as usize & 63, 0);
        self._resident = Some(buf);
    }

    /// Borrow row `i` as a `&[i8]` of length `d` (pages in from disk on first touch).
    #[inline]
    pub fn row(&self, i: usize) -> &[i8] {
        debug_assert!(i < self.nb);
        unsafe { std::slice::from_raw_parts(self.base.add(i * self.d), self.d) }
    }
}
