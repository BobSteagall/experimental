#ifndef KEWB_SIMD_HPP_DEFINED
#define KEWB_SIMD_HPP_DEFINED

#include <cstdio>
#include <cstdint>

#include <complex>
#include <limits>
#include <type_traits>

#ifdef __OPTIMIZE__
    #include <immintrin.h>
    #define KEWB_FORCE_INLINE   __attribute__((__always_inline__)) inline
#else
    #define __OPTIMIZE__
    #include <immintrin.h>
    #undef __OPTIMIZE__
    #define KEWB_FORCE_INLINE   inline
#endif

namespace simd {

using rd_128 = __m128d;
using rf_128 = __m128;
using ri_128 = __m128i;

using rd256 = __m256d;
using rf256 = __m256;
using ri256 = __m256i;
using dr256 = __m256d;
using fr256 = __m256;
using ir256 = __m256i;

using rd512 = __m512d;
using rf512 = __m512;
using ri512 = __m512i;
using dr512 = __m512d;
using fr512 = __m512;
using ir512 = __m512i;
using r512d = __m512d;
using r512f = __m512;
using r512i = __m512i;

using m256 = __m256i;
using m512 = uint32_t;

void    print_reg(char const* name, uint32_t i);
void    print_reg(char const* name, __m128d r);
void    print_reg(char const* name, __m128  r);
void    print_reg(char const* name, __m128i r);
void    print_reg(char const* name, __m256d r);
void    print_reg(char const* name, __m256  r);
void    print_reg(char const* name, __m256i r);
void    print_reg(char const* name, __m512d r);
void    print_reg(char const* name, __m512  r);
void    print_reg(char const* name, __m512i r);
void    print_mask(char const* name, uint32_t mask, int bits);
void    print_mask(char const* name, __m256i mask, int);

#define PRINT_REG(R)        print_reg(#R, R)
#define PRINT_MASK(M)       print_mask(#M, M, 16)
#define PRINT_MASK8(M)      print_mask(#M, M, 8)
#define PRINT_MASK16(M)     print_mask(#M, M, 16)
#define PRINT_LINE()        printf("\n");


KEWB_FORCE_INLINE __m512
load_value(float v)
{
    return _mm512_set1_ps(v);
}

KEWB_FORCE_INLINE __m512i
load_value(int32_t i)
{
    return _mm512_set1_epi32(i);
}

KEWB_FORCE_INLINE r512f
load_from(float const* psrc)
{
    return _mm512_loadu_ps(psrc);
}

KEWB_FORCE_INLINE r512f
masked_load_from(float const* psrc, float fill, m512 mask)
{
    return _mm512_mask_loadu_ps(_mm512_set1_ps(fill), (__mmask16) mask, psrc);
}

KEWB_FORCE_INLINE r512f
masked_load_from(float const* psrc, r512f fill, m512 mask)
{
    return _mm512_mask_loadu_ps(fill, (__mmask16) mask, psrc);
}


KEWB_FORCE_INLINE void
store_to(float* pdst, __m512 r)
{
    _mm512_mask_storeu_ps(pdst, (__mmask16) 0xFFFFu, r);
}

KEWB_FORCE_INLINE void
masked_store_to(float* pdst, __m512 r, m512 mask)
{
    _mm512_mask_storeu_ps(pdst, (__mmask16) mask, r);
}


//- Functions for moving values around
//
template<unsigned A=0, unsigned B=0, unsigned C=0, unsigned D=0,
         unsigned E=0, unsigned F=0, unsigned G=0, unsigned H=0,
         unsigned I=0, unsigned J=0, unsigned K=0, unsigned L=0,
         unsigned M=0, unsigned N=0, unsigned O=0, unsigned P=0>
KEWB_FORCE_INLINE constexpr uint32_t
make_bit_mask()
{
    static_assert((A < 2) && (B < 2) && (C < 2) && (D < 2) &&
                  (E < 2) && (F < 2) && (G < 2) && (H < 2) &&
                  (I < 2) && (J < 2) && (K < 2) && (L < 2) &&
                  (M < 2) && (N < 2) && (O < 2) && (P < 2));

    return ((A <<  0) | (B <<  1) | (C <<  2) | (D <<  3) |
            (E <<  4) | (F <<  5) | (G <<  6) | (H <<  7) |
            (I <<  8) | (J <<  9) | (K << 10) | (L << 11) |
            (M << 12) | (N << 13) | (O << 14) | (P << 15));
}

KEWB_FORCE_INLINE __m512
blend(__m512 r0, __m512 r1, uint32_t mask)
{
    return _mm512_mask_blend_ps((__mmask16) mask, r0, r1);
}


KEWB_FORCE_INLINE __m512
permute(__m512 r, __m512i perm)
{
    return _mm512_permutexvar_ps(perm, r);
}

KEWB_FORCE_INLINE __m512
masked_permute(__m512 r0, __m512 r1, __m512i perm, uint32_t mask)
{
    return _mm512_mask_permutexvar_ps(r0, (__mmask16) mask, perm, r1);
}

template<unsigned A, unsigned B, unsigned C, unsigned D,
         unsigned E, unsigned F, unsigned G, unsigned H,
         unsigned I, unsigned J, unsigned K, unsigned L,
         unsigned M, unsigned N, unsigned O, unsigned P>
KEWB_FORCE_INLINE __m512i
make_perm_map()
{
    static_assert((A < 16) && (B < 16) && (C < 16) && (D < 16) &&
                  (E < 16) && (F < 16) && (G < 16) && (H < 16) &&
                  (I < 16) && (J < 16) && (K < 16) && (L < 16) &&
                  (M < 16) && (N < 16) && (O < 16) && (P < 16));

    return _mm512_setr_epi32(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P);
}

template<int R>
KEWB_FORCE_INLINE __m512
rotate(__m512 r0)
{
    if constexpr ((R % 16) == 0)
    {
        return r0;
    }
    else
    {
        constexpr int    S = (R > 0) ? (16 - (R % 16)) : -R;
        constexpr int    A = (S + 0) % 16;
        constexpr int    B = (S + 1) % 16;
        constexpr int    C = (S + 2) % 16;
        constexpr int    D = (S + 3) % 16;
        constexpr int    E = (S + 4) % 16;
        constexpr int    F = (S + 5) % 16;
        constexpr int    G = (S + 6) % 16;
        constexpr int    H = (S + 7) % 16;
        constexpr int    I = (S + 8) % 16;
        constexpr int    J = (S + 9) % 16;
        constexpr int    K = (S + 10) % 16;
        constexpr int    L = (S + 11) % 16;
        constexpr int    M = (S + 12) % 16;
        constexpr int    N = (S + 13) % 16;
        constexpr int    O = (S + 14) % 16;
        constexpr int    P = (S + 15) % 16;

        return _mm512_permutexvar_ps(_mm512_setr_epi32(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P), r0);
    }
}

template<int R>
KEWB_FORCE_INLINE __m512
rotate_down(__m512 r0)
{
    static_assert(R >= 0);
    return rotate<-R>(r0);
}

template<int R>
KEWB_FORCE_INLINE __m512
rotate_up(__m512 r0)
{
    static_assert(R >= 0);
    return rotate<R>(r0);
}

template<int S>
KEWB_FORCE_INLINE constexpr uint32_t
shift_down_blend_mask()
{
    static_assert(S >= 0  &&  S <= 16);
    return (0xFFFFu << (unsigned)(16 - S)) & 0xFFFFu;
}


template<int S>
KEWB_FORCE_INLINE __m512
shift_down(__m512 r0)
{
    return blend(rotate_down<S>(r0), load_value(0), shift_down_blend_mask<S>());
}

template<int S>
KEWB_FORCE_INLINE __m512
shift_down_with_carry(__m512 lo, __m512 hi)
{
    return blend(rotate_down<S>(lo), rotate_down<S>(hi), shift_down_blend_mask<S>());
}

template<int S>
KEWB_FORCE_INLINE constexpr uint32_t
shift_up_blend_mask()
{
    static_assert(S >= 0  &&  S <= 16);
    return (0xFFFFu << (unsigned) S) & 0xFFFFu;
}

template<int S>
KEWB_FORCE_INLINE __m512
shift_up(__m512 r0)
{
    return blend(rotate_up<S>(r0), load_value(0), shift_up_blend_mask<S>());
}
template<int S>
KEWB_FORCE_INLINE __m512
shift_up_with_carry(__m512 lo, __m512 hi)
{
    return blend(rotate_up<S>(lo), rotate_up<S>(hi), shift_up_blend_mask<S>());
}

template<int BIAS, uint32_t MASK>
KEWB_FORCE_INLINE __m512i
make_shift_permutation()
{
    constexpr int32_t   a = ((BIAS + 0)  % 16) | ((MASK & 1u)        ? 0x10 : 0);
    constexpr int32_t   b = ((BIAS + 1)  % 16) | ((MASK & 1u << 1u)  ? 0x10 : 0);
    constexpr int32_t   c = ((BIAS + 2)  % 16) | ((MASK & 1u << 2u)  ? 0x10 : 0);
    constexpr int32_t   d = ((BIAS + 3)  % 16) | ((MASK & 1u << 3u)  ? 0x10 : 0);
    constexpr int32_t   e = ((BIAS + 4)  % 16) | ((MASK & 1u << 4u)  ? 0x10 : 0);
    constexpr int32_t   f = ((BIAS + 5)  % 16) | ((MASK & 1u << 5u)  ? 0x10 : 0);
    constexpr int32_t   g = ((BIAS + 6)  % 16) | ((MASK & 1u << 6u)  ? 0x10 : 0);
    constexpr int32_t   h = ((BIAS + 7)  % 16) | ((MASK & 1u << 7u)  ? 0x10 : 0);
    constexpr int32_t   i = ((BIAS + 8)  % 16) | ((MASK & 1u << 8u)  ? 0x10 : 0);
    constexpr int32_t   j = ((BIAS + 9)  % 16) | ((MASK & 1u << 9u)  ? 0x10 : 0);
    constexpr int32_t   k = ((BIAS + 10) % 16) | ((MASK & 1u << 10u) ? 0x10 : 0);
    constexpr int32_t   l = ((BIAS + 11) % 16) | ((MASK & 1u << 11u) ? 0x10 : 0);
    constexpr int32_t   m = ((BIAS + 12) % 16) | ((MASK & 1u << 12u) ? 0x10 : 0);
    constexpr int32_t   n = ((BIAS + 13) % 16) | ((MASK & 1u << 13u) ? 0x10 : 0);
    constexpr int32_t   o = ((BIAS + 14) % 16) | ((MASK & 1u << 14u) ? 0x10 : 0);
    constexpr int32_t   p = ((BIAS + 15) % 16) | ((MASK & 1u << 15u) ? 0x10 : 0);

    return _mm512_setr_epi32(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p);
}

template<int S>
KEWB_FORCE_INLINE void
in_place_shift_down_with_carry(__m512& lo, __m512& hi)
{
    static_assert(S >= 0  &&  S <= 16);

    constexpr uint32_t  zmask = (0xFFFFu >> (unsigned) S);
    constexpr uint32_t  bmask = ~zmask & 0xFFFFu;
    __m512i             perm  = make_shift_permutation<S, bmask>();

    lo = _mm512_permutex2var_ps(lo, perm, hi);
    hi = _mm512_maskz_permutex2var_ps((__mmask16) zmask, hi, perm, hi);
}


//- Functions for arithmetic
//
KEWB_FORCE_INLINE __m512
minimum(__m512 r0, __m512 r1)
{
    return _mm512_min_ps(r0, r1);
}

__m512
KEWB_FORCE_INLINE maximum(__m512 r0, __m512 r1)
{
    return _mm512_max_ps(r0, r1);
}

KEWB_FORCE_INLINE __m512
fused_multiply_add(__m512 r0, __m512 r1, __m512 acc)
{
    return _mm512_fmadd_ps(r0, r1, acc);
}


template<int KernelSize, int KernelCenter>
void
avx_convolve(float* pDst, float const* pKrnl, float const* pSrc, size_t len)
{
    //- The convolution kernel must have non-negative size and fit with a single reister.
    //
    static_assert(KernelSize > 1  &&  KernelSize <= 16);

    //- Thie index of the kernel center must be valid.
    //
    static_assert(KernelCenter >= 0  &&  KernelCenter < KernelSize);

    //- Convolution flips the kernel, and so the kernel center (in kernel array coordinates)
    //  must be converted to the coordinates of the window.
    //
    constexpr int   windowCenter = KernelSize - KernelCenter - 1;

    __m512  prev;   //- Bottom of the input data window
    __m512  curr;   //- Middle of the input data windows
    __m512  next;   //- Top of the input data window
    __m512  lo;     //- Primary work data register, used to multiply kernel coefficients
    __m512  hi;     //- Upper work data register, supplies values to the top of 'lo'
    __m512  sum;    //- Accumulated value

    __m512  kcoeff[KernelSize];     //- Coefficients of the convolution kernel

    //- Broadcast each kernel coefficient into its own register, to be used later in the FMA call.
    //
    for (int i = 0, j = KernelSize - 1;  i < KernelSize;  ++i, --j)
    {
        kcoeff[i] = load_value(pKrnl[j]);
    }

    //- Preload the initial input data window; note the zeroes in the register representing data
    //  preceding the input array.
    //
    prev = load_value(0.0f);
    curr = load_from(pSrc);
    next = load_from(pSrc + 16);

    for (auto pEnd = pSrc + len - 16;  pSrc < pEnd;  pSrc += 16, pDst += 16)
    {
        sum = load_value(0.0f);

        //- Init the work data registers to the correct offset in the input data window.
        //
        lo = shift_up_with_carry<windowCenter>(prev, curr);
        hi = shift_up_with_carry<windowCenter>(curr, next);

        //- Slide the input data window upward by a register's work of values.  This
        //  could also be done at the bottom of the loop, but experimentation has shown
        //  that sliding the input data window here results in slightly better performance.
        //
        prev = curr;
        curr = next;
        next = load_from(pSrc + 32);

        for (int k = 0;  k < KernelSize;  ++k)
        {
            sum = fused_multiply_add(kcoeff[k], lo, sum);   //- Update the accumulator
            in_place_shift_down_with_carry<1>(lo, hi);
        }

        store_to(pDst, sum);
    }
}



KEWB_FORCE_INLINE __m512
compare_with_exchange(__m512 vals, __m512i perm, uint32_t mask)
{
    __m512  exch = permute(vals, perm);
    __m512  vmin = minimum(vals, exch);
    __m512  vmax = maximum(vals, exch);

    return blend(vmin, vmax, mask);
}

KEWB_FORCE_INLINE __m512
sort_two_lanes_of_8(rf512 vals)
{
    //- Precompute the permutations and bitmasks for the 6 stages of this bitonic sorting sequence.
    //                                    0   1   2   3   4   5   6   7     0   1   2   3   4   5   6   7
    //                                   ------------------------------------------------------------------
    ri512 const     perm0 = make_perm_map<1,  0,  3,  2,  5,  4,  7,  6,    9,  8, 11, 10, 13, 12, 15, 14>();
    constexpr m512  mask0 = make_bit_mask<0,  1,  0,  1,  0,  1,  0,  1,    0,  1,  0,  1,  0,  1,  0,  1>();

    ri512 const     perm1 = make_perm_map<3,  2,  1,  0,  7,  6,  5,  4,   11, 10,  9,  8, 15, 14, 13, 12>();
    constexpr m512  mask1 = make_bit_mask<0,  0,  1,  1,  0,  0,  1,  1,    0,  0,  1,  1,  0,  0,  1,  1>();

    ri512 const     perm2 = make_perm_map<1,  0,  3,  2,  5,  4,  7,  6,    9,  8, 11, 10, 13, 12, 15, 14>();
    constexpr m512  mask2 = make_bit_mask<0,  1,  0,  1,  0,  1,  0,  1,    0,  1,  0,  1,  0,  1,  0,  1>();

    ri512 const     perm3 = make_perm_map<7,  6,  5,  4,  3,  2,  1,  0,   15, 14, 13, 12, 11, 10,  9,  8>();
    constexpr m512  mask3 = make_bit_mask<0,  0,  0,  0,  1,  1,  1,  1,    0,  0,  0,  0,  1,  1,  1,  1>();

    ri512 const     perm4 = make_perm_map<2,  3,  0,  1,  6,  7,  4,  5,   10, 11,  8,  9, 14, 15, 12, 13>();
    constexpr m512  mask4 = make_bit_mask<0,  0,  1,  1,  0,  0,  1,  1,    0,  0,  1,  1,  0,  0,  1,  1>();

    ri512 const     perm5 = make_perm_map<1,  0,  3,  2,  5,  4,  7,  6,    9,  8, 11, 10, 13, 12, 15, 14>();
    constexpr m512  mask5 = make_bit_mask<0,  1,  0,  1,  0,  1,  0,  1,    0,  1,  0,  1,  0,  1,  0,  1>();

    vals = compare_with_exchange(vals, perm0, mask0);
    vals = compare_with_exchange(vals, perm1, mask1);
    vals = compare_with_exchange(vals, perm2, mask2);
    vals = compare_with_exchange(vals, perm3, mask3);
    vals = compare_with_exchange(vals, perm4, mask4);
    vals = compare_with_exchange(vals, perm5, mask5);

    return vals;
}

KEWB_FORCE_INLINE __m512
sort_two_lanes_of_7(rf512 vals)
{
    //- Precompute the permutations and bitmasks for the 6 stages of this bitonic sorting sequence.
    //                                    0   1   2   3   4   5   6   7     0   1   2   3   4   5   6   7
    //                                    ---------------------------------------------------------------
    ri512 const     perm0 = make_perm_map<4,  5,  6,  3,  0,  1,  2,  7,   12, 13, 14, 11,  8,  9, 10, 15>();
    constexpr m512  mask0 = make_bit_mask<0,  0,  0,  0,  1,  1,  1,  0,    0,  0,  0,  0,  1,  1,  1,  0>();

    ri512 const     perm1 = make_perm_map<2,  3,  0,  1,  6,  5,  4,  7,   10, 11,  8,  9, 14, 13, 12, 15>();
    constexpr m512  mask1 = make_bit_mask<0,  0,  1,  1,  0,  0,  1,  0,   0,  0,  1,  1,  0,  0,  1,  0>();

    ri512 const     perm2 = make_perm_map<1,  0,  4,  5,  2,  3,  6,  7,    9,  8, 12, 13, 10, 11, 14, 15>();
    constexpr m512  mask2 = make_bit_mask<0,  1,  0,  0,  1,  1,  0,  0,    0,  1,  0,  0,  1,  1,  0,  0>();

    ri512 const     perm3 = make_perm_map<0,  1,  3,  2,  5,  4,  6,  7,    8,  9, 11, 10, 13, 12, 14, 15>();
    constexpr m512  mask3 = make_bit_mask<0,  0,  0,  1,  0,  1,  0,  0,    0,  0,  0,  1,  0,  1,  0,  0>();

    ri512 const     perm4 = make_perm_map<0,  4,  2,  6,  1,  5,  3,  7,    8, 12, 10, 14,  9, 13, 11, 15>();
    constexpr m512  mask4 = make_bit_mask<0,  0,  0,  0,  1,  0,  1,  0,    0,  0,  0,  0,  1,  0,  1,  0>();

    ri512 const     perm5 = make_perm_map<0,  2,  1,  4,  3,  6,  5,  7,    8, 10,  9, 12, 11, 14, 13, 15>();
    constexpr m512  mask5 = make_bit_mask<0,  0,  1,  0,  1,  0,  1,  0,    0,  0,  1,  0,  1,  0,  1,  0>();

    vals = compare_with_exchange(vals, perm0, mask0);
    vals = compare_with_exchange(vals, perm1, mask1);
    vals = compare_with_exchange(vals, perm2, mask2);
    vals = compare_with_exchange(vals, perm3, mask3);
    vals = compare_with_exchange(vals, perm4, mask4);
    vals = compare_with_exchange(vals, perm5, mask5);

    return vals;
}

void    avx_median_of_7(float* pdst, float const* psrc, size_t const buf_len);
void    avx_symm_convolve(float* pdst, float const* pkrnl, size_t klen, float const* psrc, size_t len);


KEWB_FORCE_INLINE __m512
load_values(float a, float b, float c, float d, float e, float f, float g, float h,
            float i, float j, float k, float l, float m, float n, float o, float p)
{
    return _mm512_setr_ps(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
}

template<int A, int B, int C, int D, int E, int F, int G, int H,
         int I, int J, int K, int L, int M, int N, int O, int P>
KEWB_FORCE_INLINE __m512i
load_values()
{
    return _mm512_setr_epi32(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
}

KEWB_FORCE_INLINE __m512i
load_values(int a, int b, int c, int d, int e, int f, int g, int h,
            int i, int j, int k, int l, int m, int n, int o, int p)
{
    return _mm512_setr_epi32(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
}

template<int A, int B, int C, int D, int E, int F, int G, int H,
         int I, int J, int K, int L, int M, int N, int O, int P>
KEWB_FORCE_INLINE __m512i
permute(__m512i r0)
{
    return _mm512_permutexvar_epi32(_mm512_setr_epi32(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P), r0);
}



#if 0
KEWB_FORCE_INLINE __m256
sort_two_lanes_of_8(rf256 vals)
{
    //- Precompute the permutations and bitmasks for the 6 stages of this bitonic sorting sequence.
    //                                   0   1   2   3   4   5   6   7
    //                                  ---------------------------------------------------
    ri256 const     perm0 = make_perm_map<1,  0,  3,  2,  5,  4,  7,  6>();
    constexpr m256  mask0 = make_bit_mask<0,  1,  0,  1,  0,  1,  0,  1>();

    ri256 const     perm1 = make_perm_map<3,  2,  1,  0,  7,  6,  5,  4>();
    constexpr m256  mask1 = make_bit_mask<0,  0,  1,  1,  0,  0,  1,  1>();

    ri256 const     perm2 = make_perm_map<1,  0,  3,  2,  5,  4,  7,  6>();
    constexpr m256  mask2 = make_bit_mask<0,  1,  0,  1,  0,  1,  0,  1>();

    ri256 const     perm3 = make_perm_map<7,  6,  5,  4,  3,  2,  1,  0>();
    constexpr m256  mask3 = make_bit_mask<0,  0,  0,  0,  1,  1,  1,  1>();

    ri256 const     perm4 = make_perm_map<2,  3,  0,  1,  6,  7,  4,  5>();
    constexpr m256  mask4 = make_bit_mask<0,  0,  1,  1,  0,  0,  1,  1>();

    ri256 const     perm5 = make_perm_map<1,  0,  3,  2,  5,  4,  7,  6>();
    constexpr m256  mask5 = make_bit_mask<0,  1,  0,  1,  0,  1,  0,  1>();

    vals = compare_with_exchange(vals, perm0, mask0);
    vals = compare_with_exchange(vals, perm1, mask1);
    vals = compare_with_exchange(vals, perm2, mask2);
    vals = compare_with_exchange(vals, perm3, mask3);
    vals = compare_with_exchange(vals, perm4, mask4);
    vals = compare_with_exchange(vals, perm5, mask5);

    return vals;
}

KEWB_FORCE_INLINE __m256
sort_two_lanes_of_7(rf256 vals)
{
    //- Precompute the permutations and bitmasks for the 6 stages of this bitonic sorting sequence.
    //                                   0   1   2   3   4   5   6   7
    //                                  ---------------------------------------------------
    ri256 const     perm0 = make_perm_map<4,  5,  6,  3,  0,  1,  2,  7>();
    constexpr m256  mask0 = make_bit_mask<0,  0,  0,  0,  1,  1,  1,  0>();

    ri256 const     perm1 = make_perm_map<2,  3,  0,  1,  6,  5,  4,  7>();
    constexpr m256  mask1 = make_bit_mask<0,  0,  1,  1,  0,  0,  1,  0>();

    ri256 const     perm2 = make_perm_map<1,  0,  4,  5,  2,  3,  6,  7>();
    constexpr m256  mask2 = make_bit_mask<0,  1,  0,  0,  1,  1,  0,  0>();

    ri256 const     perm3 = make_perm_map<0,  1,  3,  2,  5,  4,  6,  7>();
    constexpr m256  mask3 = make_bit_mask<0,  0,  0,  1,  0,  1,  0,  0>();

    ri256 const     perm4 = make_perm_map<0,  4,  2,  6,  1,  5,  3,  7>();
    constexpr m256  mask4 = make_bit_mask<0,  0,  0,  0,  1,  0,  1,  0>();

    ri256 const     perm5 = make_perm_map<0,  2,  1,  4,  3,  6,  5,  7>();
    constexpr m256  mask5 = make_bit_mask<0,  0,  1,  0,  1,  0,  1,  0>();

    vals = compare_with_exchange(vals, perm0, mask0);
    vals = compare_with_exchange(vals, perm1, mask1);
    vals = compare_with_exchange(vals, perm2, mask2);
    vals = compare_with_exchange(vals, perm3, mask3);
    vals = compare_with_exchange(vals, perm4, mask4);
    vals = compare_with_exchange(vals, perm5, mask5);

    return vals;
}

KEWB_FORCE_INLINE __m512
load_from_address(void const* psrc)
{
    return _mm512_loadu_ps(psrc);
}
template<int S>
KEWB_FORCE_INLINE __m512
shift_up_and_fill(__m512 r0, float fill)
{
    return blend(rotate_up<S>(r0), load_value(fill), shift_up_blend_mask<S>());
}

template<int S>
KEWB_FORCE_INLINE __m512
shift_down_and_fill(__m512 r0, float fill)
{
    return blend(rotate_down<S>(r0), load_value(fill), shift_down_blend_mask<S>());
}


template<unsigned... IDXS>
KEWB_FORCE_INLINE auto
make_perm_map2()
{
    static_assert(sizeof...(IDXS) == 8  ||  sizeof...(IDXS) == 16);

    if constexpr (sizeof...(IDXS) == 8)
    {
        return _mm256_setr_epi32(IDXS...);
    }
    else
    {
        return load_values<IDXS...>();
    }
}

KEWB_FORCE_INLINE __m512
mask_permute2(__m512 r0, __m512 r1, __m512i perm, uint32_t mask)
{
    return _mm512_mask_permutex2var_ps(r0, (__mmask16) mask, perm, r1);
}

KEWB_FORCE_INLINE r512d
load_from(double const* psrc)
{
    return _mm512_loadu_pd(psrc);
}

KEWB_FORCE_INLINE r512f
load_from(cx_float const* psrc)
{
    return _mm512_loadu_ps(psrc);
}
KEWB_FORCE_INLINE __m512i
permute(__m512i r, __m512i perm)
{
    return _mm512_permutexvar_epi32(perm, r);
}

KEWB_FORCE_INLINE r512i
load_from(int32_t const* psrc)
{
    return _mm512_loadu_epi32(psrc);
}

KEWB_FORCE_INLINE void
store_to(int32_t* pdst, __m512i r)
{
    _mm512_mask_storeu_epi32(pdst, (__mmask16) 0xFFFFu, r);
}

KEWB_FORCE_INLINE void
masked_store_to(int32_t* pdst, __m512i r, m512 mask)
{
    _mm512_mask_storeu_epi32(pdst, (__mmask16) mask, r);
}

template<int R>
KEWB_FORCE_INLINE __m512i
rotate(__m512i r0)
{
    if constexpr ((R % 16) == 0)
    {
        return r0;
    }
    else
    {
        constexpr int    S = (R > 0) ? (16 - (R % 16)) : -R;
        constexpr int    A = (S + 0) % 16;
        constexpr int    B = (S + 1) % 16;
        constexpr int    C = (S + 2) % 16;
        constexpr int    D = (S + 3) % 16;
        constexpr int    E = (S + 4) % 16;
        constexpr int    F = (S + 5) % 16;
        constexpr int    G = (S + 6) % 16;
        constexpr int    H = (S + 7) % 16;
        constexpr int    I = (S + 8) % 16;
        constexpr int    J = (S + 9) % 16;
        constexpr int    K = (S + 10) % 16;
        constexpr int    L = (S + 11) % 16;
        constexpr int    M = (S + 12) % 16;
        constexpr int    N = (S + 13) % 16;
        constexpr int    O = (S + 14) % 16;
        constexpr int    P = (S + 15) % 16;

        return _mm512_permutexvar_epi32(_mm512_setr_epi32(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P), r0);
    }
}

using cx_float = std::complex<float>;



#endif

}       //- simd namespace
#endif  //- KEWB_SIMD_HPP_DEFINED
