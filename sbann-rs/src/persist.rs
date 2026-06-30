//! On-disk index serialization (SBANN_INDEX_SAVE / SBANN_INDEX_LOAD).
//!
//! Build-once / load-fast: a full `Index` (POD arrays + the concrete router/compressor) is written
//! to a single file with an 8-byte magic + u32 version header. Trait objects are NOT serialized
//! generically — `vq.rs` matches the concrete `Router`/`Compressor` type, writes a 1-byte type tag,
//! then its POD fields. Loading reconstructs the same concrete type from those fields.
//!
//! WRITE side streams straight to a `BufWriter<File>` (low RAM — the n*a0*d `raw`/`blocks` arrays
//! never get a second in-memory copy). READ side mmaps the file and COPIES each length-prefixed
//! region into an owned `Vec` (the `Index` owns its arrays), then drops the mmap. All multi-byte
//! integers are little-endian; vector regions are length-prefixed (u64 byte count). Reads decode
//! u32/i32/f32 element-by-element so they are immune to the mmap's arbitrary alignment.

use std::io::{self, Write};

pub const MAGIC: &[u8; 8] = b"SBANNIX1";
pub const VERSION: u32 = 1;

/// Streaming writer over any `Write` (we use `BufWriter<File>`). Every method appends little-endian.
pub struct Sw<'a> {
    pub w: &'a mut dyn Write,
}

impl<'a> Sw<'a> {
    #[inline]
    pub fn u8(&mut self, v: u8) -> io::Result<()> { self.w.write_all(&[v]) }
    #[inline]
    pub fn u32(&mut self, v: u32) -> io::Result<()> { self.w.write_all(&v.to_le_bytes()) }
    #[inline]
    pub fn u64(&mut self, v: u64) -> io::Result<()> { self.w.write_all(&v.to_le_bytes()) }
    #[inline]
    pub fn usize(&mut self, v: usize) -> io::Result<()> { self.u64(v as u64) }
    #[inline]
    pub fn f32(&mut self, v: f32) -> io::Result<()> { self.w.write_all(&v.to_le_bytes()) }
    /// length-prefixed raw byte region (the building block for every typed-slice write).
    #[inline]
    pub fn bytes(&mut self, b: &[u8]) -> io::Result<()> { self.u64(b.len() as u64)?; self.w.write_all(b) }
    pub fn u8s(&mut self, v: &[u8]) -> io::Result<()> { self.bytes(v) }
    pub fn i8s(&mut self, v: &[i8]) -> io::Result<()> { self.bytes(bytemuck::cast_slice(v)) }
    pub fn u32s(&mut self, v: &[u32]) -> io::Result<()> { self.bytes(bytemuck::cast_slice(v)) }
    pub fn i32s(&mut self, v: &[i32]) -> io::Result<()> { self.bytes(bytemuck::cast_slice(v)) }
    pub fn f32s(&mut self, v: &[f32]) -> io::Result<()> { self.bytes(bytemuck::cast_slice(v)) }
    /// `Vec<usize>` (e.g. router beams): count then each element as u64.
    pub fn usizes(&mut self, v: &[usize]) -> io::Result<()> {
        self.u64(v.len() as u64)?;
        for &x in v { self.u64(x as u64)?; }
        Ok(())
    }
}

/// Cursor reader over an in-memory (mmap'd) byte buffer. Decodes the same layout `Sw` wrote.
pub struct Pr<'a> {
    pub buf: &'a [u8],
    pub pos: usize,
}

impl<'a> Pr<'a> {
    pub fn new(buf: &'a [u8]) -> Self { Pr { buf, pos: 0 } }
    #[inline]
    pub fn u8(&mut self) -> u8 { let v = self.buf[self.pos]; self.pos += 1; v }
    #[inline]
    pub fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4; v
    }
    #[inline]
    pub fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8; v
    }
    #[inline]
    pub fn usize(&mut self) -> usize { self.u64() as usize }
    #[inline]
    pub fn f32(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4; v
    }
    /// borrow a length-prefixed byte region from the underlying buffer (no copy).
    #[inline]
    pub fn bytes(&mut self) -> &'a [u8] {
        let n = self.usize();
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n; s
    }
    pub fn u8_vec(&mut self) -> Vec<u8> { self.bytes().to_vec() }
    pub fn i8_vec(&mut self) -> Vec<i8> { bytemuck::cast_slice::<u8, i8>(self.bytes()).to_vec() }
    // u32/i32/f32 decode element-by-element: the mmap offset is not 4-byte aligned in general, so a
    // direct bytemuck::cast_slice would panic. chunks_exact + from_le_bytes is alignment-independent.
    pub fn u32_vec(&mut self) -> Vec<u32> {
        self.bytes().chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    pub fn i32_vec(&mut self) -> Vec<i32> {
        self.bytes().chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    pub fn f32_vec(&mut self) -> Vec<f32> {
        self.bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    pub fn usize_vec(&mut self) -> Vec<usize> {
        let n = self.usize();
        (0..n).map(|_| self.usize()).collect()
    }
}
