#ifndef KEWB_SIMD_HPP_DEFINED
#define KEWB_SIMD_HPP_DEFINED

#include <cstdio>
#include <cstdint>

#include <type_traits>

#ifdef __OPTIMIZE__
    #include <immintrin.h>
    #define KEWB_FORCE_INLINE   inline __attribute__((__always_inline__))
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

using rd_256  = __m256d;
using rf_256  = __m256;
using ri_256  = __m256i;
using msk_256 = __m256i;

using rd_512  = __m512d;
using rf_512  = __m512;
using ri_512  = __m512i;
using msk_512 = uint32_t;


void    print_reg(char const* name, uint32_t i);
void    print_reg(char const* name, rd_128 r);
void    print_reg(char const* name, rf_128  r);
void    print_reg(char const* name, ri_128 r);
void    print_reg(char const* name, rd_256 r);
void    print_reg(char const* name, rf_256  r);
void    print_reg(char const* name, ri_256 r);
void    print_reg(char const* name, rd_512 r);
void    print_reg(char const* name, rf_512  r);
void    print_reg(char const* name, ri_512 r);
void    print_mask(char const* name, uint32_t mask, int bits);
void    print_mask(char const* name, ri_256 mask, int);

#define PRINT_REG(R)        print_reg(#R, R)
#define PRINT_MASK(M)       print_mask(#M, M, 16)
#define PRINT_MASK8(M)      print_mask(#M, M, 8)
#define PRINT_MASK16(M)     print_mask(#M, M, 16)
#define PRINT_LINE()        printf("\n");


KEWB_FORCE_INLINE rf_512
load_value(float v)
{
    return _mm512_set1_ps(v);
}

KEWB_FORCE_INLINE ri_512
load_value(int32_t i)
{
    return _mm512_set1_epi32(i);
}

KEWB_FORCE_INLINE rf_512
load_from(float const* psrc)
{
    return _mm512_loadu_ps(psrc);
}

KEWB_FORCE_INLINE rf_512
masked_load_from(float const* psrc, float fill, msk_512 mask)
{
    return _mm512_mask_loadu_ps(_mm512_set1_ps(fill), (__mmask16) mask, psrc);
}

KEWB_FORCE_INLINE rf_512
masked_load_from(float const* psrc, rf_512 fill, msk_512 mask)
{
    return _mm512_mask_loadu_ps(fill, (__mmask16) mask, psrc);
}

KEWB_FORCE_INLINE rf_512
load_upper_element(rf_512 src)
{
    return _mm512_permutexvar_ps(_mm512_set1_epi32(15), src);
};

KEWB_FORCE_INLINE void
store_to(float* pdst, rf_512 r)
{
    _mm512_mask_storeu_ps(pdst, (__mmask16) 0xFFFFu, r);
}

KEWB_FORCE_INLINE void
masked_store_to(float* pdst, rf_512 r, msk_512 mask)
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

KEWB_FORCE_INLINE rf_512
blend(rf_512 r0, rf_512 r1, uint32_t mask)
{
    return _mm512_mask_blend_ps((__mmask16) mask, r0, r1);
}


KEWB_FORCE_INLINE rf_512
permute(rf_512 r, ri_512 perm)
{
    return _mm512_permutexvar_ps(perm, r);
}

KEWB_FORCE_INLINE rf_512
masked_permute(rf_512 r0, rf_512 r1, ri_512 perm, uint32_t mask)
{
    return _mm512_mask_permutexvar_ps(r0, (__mmask16) mask, perm, r1);
}

template<unsigned A, unsigned B, unsigned C, unsigned D,
         unsigned E, unsigned F, unsigned G, unsigned H,
         unsigned I, unsigned J, unsigned K, unsigned L,
         unsigned M, unsigned N, unsigned O, unsigned P>
KEWB_FORCE_INLINE ri_512
make_perm_map()
{
    static_assert((A < 16) && (B < 16) && (C < 16) && (D < 16) &&
                  (E < 16) && (F < 16) && (G < 16) && (H < 16) &&
                  (I < 16) && (J < 16) && (K < 16) && (L < 16) &&
                  (M < 16) && (N < 16) && (O < 16) && (P < 16));

    return _mm512_setr_epi32(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P);
}

template<int R>
KEWB_FORCE_INLINE rf_512
rotate(rf_512 r0)
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
KEWB_FORCE_INLINE rf_512
rotate_down(rf_512 r0)
{
    static_assert(R >= 0);
    return rotate<-R>(r0);
}

template<int R>
KEWB_FORCE_INLINE rf_512
rotate_up(rf_512 r0)
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
KEWB_FORCE_INLINE rf_512
shift_down(rf_512 r0)
{
    return blend(rotate_down<S>(r0), load_value(0), shift_down_blend_mask<S>());
}

template<int S>
KEWB_FORCE_INLINE rf_512
shift_down_with_carry(rf_512 lo, rf_512 hi)
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
KEWB_FORCE_INLINE rf_512
shift_up(rf_512 r0)
{
    return blend(rotate_up<S>(r0), load_value(0), shift_up_blend_mask<S>());
}
template<int S>
KEWB_FORCE_INLINE rf_512
shift_up_with_carry(rf_512 lo, rf_512 hi)
{
    return blend(rotate_up<S>(lo), rotate_up<S>(hi), shift_up_blend_mask<S>());
}

template<int BIAS, uint32_t MASK>
KEWB_FORCE_INLINE ri_512
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
in_place_shift_down_with_carry(rf_512& lo, rf_512& hi)
{
    static_assert(S >= 0  &&  S <= 16);

    constexpr uint32_t  zmask = (0xFFFFu >> (unsigned) S);
    constexpr uint32_t  bmask = ~zmask & 0xFFFFu;
    ri_512              perm  = make_shift_permutation<S, bmask>();

    lo = _mm512_permutex2var_ps(lo, perm, hi);
    hi = _mm512_maskz_permutex2var_ps((__mmask16) zmask, hi, perm, hi);
}


//- Functions for arithmetic
//
KEWB_FORCE_INLINE rf_512
add(rf_512 r0, rf_512 r1)
{
    return _mm512_add_ps(r0, r1);
}

KEWB_FORCE_INLINE rf_512
subtract(rf_512 r0, rf_512 r1)
{
    return _mm512_sub_ps(r0, r1);
}

KEWB_FORCE_INLINE rf_512
minimum(rf_512 r0, rf_512 r1)
{
    return _mm512_min_ps(r0, r1);
}

rf_512
KEWB_FORCE_INLINE maximum(rf_512 r0, rf_512 r1)
{
    return _mm512_max_ps(r0, r1);
}

KEWB_FORCE_INLINE rf_512
fused_multiply_add(rf_512 r0, rf_512 r1, rf_512 acc)
{
    return _mm512_fmadd_ps(r0, r1, acc);
}


KEWB_FORCE_INLINE rf_512
compare_with_exchange(rf_512 vals, ri_512 perm, uint32_t mask)
{
    rf_512  exch = permute(vals, perm);
    rf_512  vmin = minimum(vals, exch);
    rf_512  vmax = maximum(vals, exch);

    return blend(vmin, vmax, mask);
}

KEWB_FORCE_INLINE rf_512
sort_two_lanes_of_8(rf_512 vals)
{
    //- Precompute the permutations and bitmasks for the 6 stages of this bitonic sorting sequence.
    //                                        0   1   2   3   4   5   6   7     0   1   2   3   4   5   6   7
    //                                       ------------------------------------------------------------------
    ri_512 const        perm0 = make_perm_map<1,  0,  3,  2,  5,  4,  7,  6,    9,  8, 11, 10, 13, 12, 15, 14>();
    constexpr msk_512   mask0 = make_bit_mask<0,  1,  0,  1,  0,  1,  0,  1,    0,  1,  0,  1,  0,  1,  0,  1>();

    ri_512 const        perm1 = make_perm_map<3,  2,  1,  0,  7,  6,  5,  4,   11, 10,  9,  8, 15, 14, 13, 12>();
    constexpr msk_512   mask1 = make_bit_mask<0,  0,  1,  1,  0,  0,  1,  1,    0,  0,  1,  1,  0,  0,  1,  1>();

    ri_512 const        perm2 = make_perm_map<1,  0,  3,  2,  5,  4,  7,  6,    9,  8, 11, 10, 13, 12, 15, 14>();
    constexpr msk_512   mask2 = make_bit_mask<0,  1,  0,  1,  0,  1,  0,  1,    0,  1,  0,  1,  0,  1,  0,  1>();

    ri_512 const        perm3 = make_perm_map<7,  6,  5,  4,  3,  2,  1,  0,   15, 14, 13, 12, 11, 10,  9,  8>();
    constexpr msk_512   mask3 = make_bit_mask<0,  0,  0,  0,  1,  1,  1,  1,    0,  0,  0,  0,  1,  1,  1,  1>();

    ri_512 const        perm4 = make_perm_map<2,  3,  0,  1,  6,  7,  4,  5,   10, 11,  8,  9, 14, 15, 12, 13>();
    constexpr msk_512   mask4 = make_bit_mask<0,  0,  1,  1,  0,  0,  1,  1,    0,  0,  1,  1,  0,  0,  1,  1>();

    ri_512 const        perm5 = make_perm_map<1,  0,  3,  2,  5,  4,  7,  6,    9,  8, 11, 10, 13, 12, 15, 14>();
    constexpr msk_512   mask5 = make_bit_mask<0,  1,  0,  1,  0,  1,  0,  1,    0,  1,  0,  1,  0,  1,  0,  1>();

    vals = compare_with_exchange(vals, perm0, mask0);
    vals = compare_with_exchange(vals, perm1, mask1);
    vals = compare_with_exchange(vals, perm2, mask2);
    vals = compare_with_exchange(vals, perm3, mask3);
    vals = compare_with_exchange(vals, perm4, mask4);
    vals = compare_with_exchange(vals, perm5, mask5);

    return vals;
}

KEWB_FORCE_INLINE rf_512
sort_two_lanes_of_7(rf_512 vals)
{
    //- Precompute the permutations and bitmasks for the 6 stages of this bitonic sorting sequence.
    //                                        0   1   2   3   4   5   6   7     0   1   2   3   4   5   6   7
    //                                       ---------------------------------------------------------------
    ri_512 const        perm0 = make_perm_map<4,  5,  6,  3,  0,  1,  2,  7,   12, 13, 14, 11,  8,  9, 10, 15>();
    constexpr msk_512   mask0 = make_bit_mask<0,  0,  0,  0,  1,  1,  1,  0,    0,  0,  0,  0,  1,  1,  1,  0>();

    ri_512 const        perm1 = make_perm_map<2,  3,  0,  1,  6,  5,  4,  7,   10, 11,  8,  9, 14, 13, 12, 15>();
    constexpr msk_512   mask1 = make_bit_mask<0,  0,  1,  1,  0,  0,  1,  0,   0,  0,  1,  1,  0,  0,  1,  0>();

    ri_512 const        perm2 = make_perm_map<1,  0,  4,  5,  2,  3,  6,  7,    9,  8, 12, 13, 10, 11, 14, 15>();
    constexpr msk_512   mask2 = make_bit_mask<0,  1,  0,  0,  1,  1,  0,  0,    0,  1,  0,  0,  1,  1,  0,  0>();

    ri_512 const        perm3 = make_perm_map<0,  1,  3,  2,  5,  4,  6,  7,    8,  9, 11, 10, 13, 12, 14, 15>();
    constexpr msk_512   mask3 = make_bit_mask<0,  0,  0,  1,  0,  1,  0,  0,    0,  0,  0,  1,  0,  1,  0,  0>();

    ri_512 const        perm4 = make_perm_map<0,  4,  2,  6,  1,  5,  3,  7,    8, 12, 10, 14,  9, 13, 11, 15>();
    constexpr msk_512   mask4 = make_bit_mask<0,  0,  0,  0,  1,  0,  1,  0,    0,  0,  0,  0,  1,  0,  1,  0>();

    ri_512 const        perm5 = make_perm_map<0,  2,  1,  4,  3,  6,  5,  7,    8, 10,  9, 12, 11, 14, 13, 15>();
    constexpr msk_512   mask5 = make_bit_mask<0,  0,  1,  0,  1,  0,  1,  0,    0,  0,  1,  0,  1,  0,  1,  0>();

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
void    integrate(int32_t length, float* pdst, float const* psrc, int32_t width);

}       //- simd namespace
#endif  //- KEWB_SIMD_HPP_DEFINED
