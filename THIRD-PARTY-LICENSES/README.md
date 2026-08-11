# Bundled Windows OCR runtime

Marginalia's universal ZIP includes a deliberately minimal Windows build of
Tesseract OCR 5.4.0.20240606 from the UB Mannheim distribution. Linux and
macOS use a separately installed system Tesseract instead.

The complete, hash-pinned runtime source and provenance are recorded in
`../OCR-RUNTIME-PROVENANCE.txt`. `../OCR-RUNTIME-FILES.txt` is the release
allowlist: the build copies no executable or DLL that is not needed by the
Tesseract command-line program used by Marginalia.

For the bundled copyleft libraries, upstream source locations are listed in
[`SOURCE-LOCATIONS.md`](SOURCE-LOCATIONS.md). That informational list is not a
project-authored offer or support commitment.

The following table maps every distributed Windows OCR file to its upstream
component and the accompanying license notice. Windows system DLLs are not
distributed and are therefore not listed.

| Distributed file(s) | Component | License notice in this directory |
| --- | --- | --- |
| `tesseract.exe`, `libtesseract-5.dll`, `tessdata/configs/quiet` | Tesseract OCR | `Tesseract-Apache-2.0.txt`, `Tesseract-AUTHORS.txt`, `Tesseract-README.md` |
| `tessdata/eng.traineddata` | Tesseract English trained data | `Tesseract-Apache-2.0.txt` |
| `libarchive-13.dll` | libarchive | `libarchive-COPYING.txt` |
| `libleptonica-6.dll` | Leptonica | `Leptonica-LICENSE.txt` |
| `libtiff-6.dll` | LibTIFF | `LibTIFF-LICENSE.md` |
| `libgcc_s_seh-1.dll`, `libstdc++-6.dll` | GCC runtime libraries | `GCC-COPYING3.txt`, `GCC-RUNTIME-LIBRARY-EXCEPTION.txt` |
| `libwinpthread-1.dll` | MinGW-w64 winpthreads | `winpthreads-COPYING.txt` |
| `libb2-1.dll` | libb2 / BLAKE2 | `libb2-COPYING.txt` |
| `libbz2-1.dll` | bzip2 | `bzip2-LICENSE.txt` |
| `libcrypto-3-x64.dll` | OpenSSL | `OpenSSL-LICENSE.txt` |
| `libexpat-1.dll` | Expat | `Expat-COPYING.txt` |
| `libiconv-2.dll` | GNU libiconv library | `libiconv-COPYING.LIB.txt` |
| `liblz4.dll` | LZ4 | `LZ4-LICENSE.txt` |
| `liblzma-5.dll` | XZ Utils / liblzma | `XZ-COPYING.txt`, `XZ-COPYING.0BSD.txt`, `XZ-COPYING.GPLv2.txt`, `XZ-COPYING.GPLv3.txt`, `XZ-COPYING.LGPLv2.1.txt` |
| `zlib1.dll` | zlib | `zlib-LICENSE.txt` |
| `libzstd.dll` | Zstandard | `zstd-LICENSE.txt` |
| `libgif-7.dll` | giflib | `giflib-COPYING.txt` |
| `libjpeg-8.dll` | libjpeg-turbo | `libjpeg-turbo-LICENSE.md` |
| `libopenjp2-7.dll` | OpenJPEG | `OpenJPEG-LICENSE.txt` |
| `libpng16-16.dll` | libpng | `libpng-LICENSE.txt` |
| `libwebp-7.dll`, `libwebpmux-3.dll`, `libsharpyuv-0.dll` | libwebp | `libwebp-COPYING.txt` |
| `libdeflate.dll` | libdeflate | `libdeflate-COPYING.txt` |
| `libjbig-0.dll` | JBIG-KIT | `JBIG-KIT-COPYING.txt` |
| `libLerc.dll` | LERC | `LERC-LICENSE.txt` |

Upstream source locations:

- Tesseract: https://github.com/tesseract-ocr/tesseract
- Tesseract trained data: https://github.com/tesseract-ocr/tessdata_fast
- Windows distribution: https://github.com/UB-Mannheim/tesseract/wiki
- libarchive: https://github.com/libarchive/libarchive
- Leptonica: https://github.com/DanBloomberg/leptonica
- LibTIFF: https://gitlab.com/libtiff/libtiff
- GCC runtime: https://gcc.gnu.org/
- MinGW-w64 winpthreads: https://www.mingw-w64.org/
- libb2: https://github.com/BLAKE2/libb2
- bzip2: https://sourceware.org/bzip2/
- OpenSSL: https://www.openssl.org/
- Expat: https://github.com/libexpat/libexpat
- GNU libiconv: https://www.gnu.org/software/libiconv/
- LZ4: https://github.com/lz4/lz4
- XZ Utils: https://tukaani.org/xz/
- zlib: https://zlib.net/
- Zstandard: https://github.com/facebook/zstd
- giflib: https://sourceforge.net/projects/giflib/
- libjpeg-turbo: https://github.com/libjpeg-turbo/libjpeg-turbo
- OpenJPEG: https://github.com/uclouvain/openjpeg
- libpng: https://github.com/pnggroup/libpng
- libwebp: https://chromium.googlesource.com/webm/libwebp/
- libdeflate: https://github.com/ebiggers/libdeflate
- JBIG-KIT: https://www.cl.cam.ac.uk/~mgk25/jbigkit/
- LERC: https://github.com/Esri/lerc

The license files are reproduced verbatim from the upstream project or from
the corresponding MSYS2/MinGW package notice. This directory is informational;
it does not change the license of Marginalia itself, which remains AGPL-3.0-only.
