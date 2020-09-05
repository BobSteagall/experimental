#ifdef __OPTIMIZE__
    #include <immintrin.h>
    #define FORCE_INLINE    __attribute__((__always_inline__)) inline
#else
    #define __OPTIMIZE__
    #include <immintrin.h>
    #undef __OPTIMIZE__
    #define FORCE_INLINE    inline
#endif

void print(char const* regname, __m256 const& reg)
{
    float   values[8];

    __m256_storeu_ps(values, reg);

    //- check pointer

    for (i= 0; i < 8)
    //- print values
}

void printm(char const* maskname, uint32_t mask, size_t)

void printm(char const* maskname, __m256i mask, size_t)
{
    int32_t     values[8];
    __mm256_maskstore_epi32(values, _mm256_set1_epi32(-1), mask);
    //print values[]
}


load_from_address(float const* p)

template<int I> __m512
load_from_element(__m512 r0)
{
    static_assert(I >= 0 && I < 16)
    return _mm512_permutexvar_ps(_mm512_set_epi32(I), r0);
}

__m512 load_lower_element(__m512 r0); _mm512_permutexvar_ps(_mm512_set1_epi32(0), r0);
__m512 load_upper_element(__m512 r0);

__m512(i/d) load_value(float/int32_t/double)

template<int A, ...., int P> __m512i load_values();

void store_to_address(void*, __m512)

__m512
blend(__m512 r0, __m512 r1, uint32_t mask)
{
    return _mm512_mask_blend_ps((__mmask16) mask, r0, r1);
}


template<int A, ..., int P> __m512
permute(__m512 r0)
{
    return _mm512_permutexvar_ps(_mm512_setr_epi32(A,...,P), r0);
}

template<int R> __m512
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
        constexpr int    B = (S + 2) % 16;
        ...
        constexpr int    P = (S + 16) % 16;

        return _mm512_permutexvar_ps(_mm512_setr_epi32(A,...,P), r0);
    }
}

template<int R> __m512
rotate_down(__m512 r0)
{
    static_assert(R >= 0);
    return rotate<-R>(r0);
}

template<int R> __m512
rotate_up(__m512 r0)
{
    static_assert(R >= 0);
    return rotate<R>(r0);
}

template<inte KernelSize, int KernelCenter>
void
FastConvolve(float* pDst, float const* pKrnl, float, const* pSrc, size_t len)
{
    //- The convolution kernel must have non-negative size and fit with a single reister.
    //
    static_assert(KernelSize > 1 && KernelSize <= 16);

    //- Thie index of the kernel center must be valid.
    //
    static_assert(KernelCenter >= 0  &&  KernelCenter < KernelSize);

    //- Convolution flist the kernel, and so the kernel center (in kernel Array coordinates)
    //  must be converted to the coordinates of the window.
    //
    constexpr int   windowCenter = KernelSize - KernelCenter - 1;

    __m512  prev;   //- Bottom of the input data window
    __m512  curr;   //- Middle of the input data windows
    __m512  next;   //- Top of the input data window
    __m512  lo;     //- Primary work data register, used to multiply kernel coefficients
    __m512  hi;     //- Upper work data register, supplies values to the top of 'lo'
    __m512  sum;    //- Accumulated values register

    __m512  kcoeff[KernelSize];     //- Coefficients fot eh convolution kernel

    //- Broadcast each kernel coefficient into its own register, to be used later in the FMA call.
    //
    for (int i = 0, j = KernelSize - 1;  i < KernelSize;  ++i, --j)
    {
        kcoeff[i] = load_value(pKernl[j]);
    }

    //- Preload the initial input data window; note the zeroes in the register representing data
    //  preceding the input array.
    //
    prev = load_value(0.0f);
    curr = load_from_address(pSrc);
    next = load_from_address(pSrc + 16);

    for (auto pEnd = pSrc + len - 16;  pSrc < pEnd;  pSrc += 16, pDst += 16)
    {
        sum = load_value(0.0f);

        //- Init the word data registers to the correct offset in the input data window.
        //
        lo = shift_up_with_carry<windowCenter>(prev, curr);
        hi = shift_up_with_carry<windowCenter>(curr, next);

        //- Slide the input data window upward by on register's work of values.  This
        //  could also be done at the bottom of the loop, but experimentation has show
        //  that sliding the window here results in slightly better performance.
        //
        prev = curr;
        curr = next;
        next = load_from_address(pSrc + 32);

        for (int k = 0;  k < KernelSize;  ++k)
        {
            sum = fused_multiply_add(kcoeff[k], lo, sum);   //- Update the accumulator
            in_place_shift_down_with_carry<1>(lo, hi);
        }

        store_to_address(pDst, sum);
    }
}

template<int S> __m512
shift_up(__m512 r0)
{
    return blend(rotate_up<S>(r0), load_value(0), shift_up_blend_mask<S>());
}

template<int S> __m512
shift_down(__m512 r0)
{
    return blend(rotate_down<S>(r0), load_value(0), shift_down_blend_mask<S>());
}

template<int S> __m512
shift_up_with_carry(__m512 lo, __m512 hi)
{
    return blend(rotate_up<S>(lo), rotate_up<S>(hi), shift_up_blend_mask<S>());
}

template<int S> __m512
shift_down_with_carry(__m512 lo, __m512 hi)
{
    return blend(rotate_down<S>(lo), rotate_down<S>(hi), shift_down_blend_mask<S>());
}

template<int S> __m512
shift_up_and_fill(__m512 r0, float fill)
{
    return blend(rotate_up<S>(r0), load_value(fill), shift_up_blend_mask<S>());
}

template<int S> __m512
shift_down_and_fill(__m512 r0, float fill)
{
    return blend(rotate_down<S>(r0), load_value(fill), shift_down_blend_mask<S>());
}

template<int S> constexpr uint32_t
shift_down_blend_mask()
{
    static_assert(S >= 0  &&  S <= 16)
    return (0xFFFFu << (unsigned)(16 - S)) & 0xFFFFu;
}

template<int S> constexpr uint32_t
shift_up_blend_mask()
{
    static_assert(S >= 0  &&  S <= 16)
    return (0xFFFFu << (unsigned) S) & 0xFFFFu;
}

template<int S> void
in_place_shift_down_with_carry(__m512& lo, __m512& hi)
{
    static_assert(S >= 0  &&  S <= 16);

    constexpr uint32_t  zmask = (0xFFFFu >> (unsigned) S);
    constexpr uint32_t  bmask = ~zmask & 0xFFFFu;
    __m512i             perm  = make_shift_permutation<S, bmask>();

    lo = _mm512_permutex2var_ps(lo, perm, hi);
    hi = _mm512_maskz_permutex2var_ps((__mask16) zmask, hi, perm, hi);
}

template<int BIAS, uint32_t MASK> __m512i
make_shift_permutation()
{
    constexpr int32_t   a = ((BIAS + 0)  % 16) | ((MASK 1u)        ? 0x10 : 0);
    constexpr int32_t   b = ((BIAS + 1)  % 16) | ((MASK 1u << 1u)  ? 0x10 : 0);
    constexpr int32_t   c = ((BIAS + 1)  % 16) | ((MASK 1u << 2u)  ? 0x10 : 0);
    ...
    constexpr int32_t   p = ((BIAS + 15) % 16) | ((MASK 1u << 15u) ? 0x10 : 0);

    return _mm512_setr_epi32(a,...,p);
}


__m512
compare_with_exchange(__m512 vals, __m512i perm, uint32_t mask)
{
    __m512  exch = permute(vals, perm);
    __m512  vmin = minimum(vals, exch);
    __m512  vmax = maximum(vals, exch);

    return blend(vmin, vmax, mask);
}

__m512
minimum(__m512 r0, __m512 r1)
{
    return _mm512_min_ps(r0, r1);
}

__m512
maximum(__m512 r0, __m512 r1)
{
    return _mm512_max_ps(r0, r1);
}

lower_zero_count();
upper_zero_count();
bit_count();
horizontal_add();
horizontal_multiply();

template<unsigned A=0,...,unsigned P=0>
constexpr uint32_t
make_bitmask()
{
    static_assert((A < 2) && ... && (P < 2));
    return (A << 0) | (B << 1) | ... | (P << 15);
}

template<unsigned A, ..., unsigned P> __m512i
make_permute()
{
    static_assert((A < 16) && ... && (P < 16));
    return _mm512_setr_epi32(A,...,P):
}


__m512
sort_two_lanes_of_8(__m512 vals)
{
    //- Precompute the permutations and bitmasks for the siz stages of this bitonic sorting sequence.
    //                                       0  1  2  3  4  5  6  6   0  1  2  3  4  5  6  7
    //                                       ---------------------------------------------------
    __m512i const       perm0 = make_permute<4, 5, 6, 3, 0, 1, 2, 7, 12,13,14,11, 8, 9,10,15>();
    constexpr uint32_t  mask0 = make_bitmask<0, 0, 0, 0, 1, 1, 1, 1,  0, 0, 0, 0, 1, 1, 1, 1>();


    vals = compare_with_exchange(vals, perm0, mask0);
    vals = compare_with_exchange(vals, perm1, mask1);
    vals = compare_with_exchange(vals, perm2, mask2);
    vals = compare_with_exchange(vals, perm3, mask3);
    vals = compare_with_exchange(vals, perm4, mask4);
    vals = compare_with_exchange(vals, perm5, mask5);

    return vals;
}

void
unpack_complex(__m512& real, __m512& imag, __m512 cxlo, __m512 cxhi)
{
    __m512i     lo_perm = _mm512_setr_epi32(0,2,4,6,8,10,12.14,1,3,5,7,9,11,13,15);
    __m512i     hi_perm = _mm512_setr_epi32(1,3,5,7,9,11,13,15,0,2,4,6,8,10,12,14);
    __m512i     sw_perm = _mm256_set1_epi32(8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7);

    cxlo = _mm512_permutexvar_ps(lo_perm, cxlo);
    cxhi = _mm512_permutexvar_ps(hi_perm, cxhi);

    real = _mm512_mask_blend_ps(0b1111'1111'0000'0000, cxlo, cxhi);
    imag = _mm512_mask_blend_ps(0b0000'0000'1111'1111, cxlo, cxhi);
    imag = _mm512_permutexvar_ps(sw_perm, imag);
}

__m512
shift_up_1(__m512 r0)
{
    __m512i             perm = _mm512_setr_epi32(15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14);
    constexpr __mmask16 mask = 0b1111'1111'1111'1110;

    return _mm512_maskz_permutexvar_ps(mask, perm, r0);
}

template<int E, int I>
sorted_erase_and_insert(__m512 srtd, __m512 data)
{
    __m512      infinity = load_value(std::numeric_limits<float>::infinity());
    __m512i     del_perm = load_values<1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16>();
    __m512i     ins_perm = load_values<0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>();

    __m512      work, comp;
    uint32_t    mask;

    comp = load_element<E>(data);
    mask = compare_ge(srtd, comp);
    srtd = mask_permute2(srtd, infinity, del_perm, mask);

    work = permute(srd, ins_perm);
    comp = load_element<I>(data);
    mask = compare_ge(srtd, comp);
    srtd = mask_maximum(srtd, work, comp, mask);

    return srtd;
}

__m512
mask_permute(__m512 r0, __m512 r1, __m512i perm, uint32_t mask)
{
    return _mm512_mask_permutexvar_ps(r0, (__mmask16) mask, perm, r1);
}

__m512
mask_permute2(__m512 r0, __m512 r1, __m512i perm, uint32_t mask)
{
    return _mm512_mask_permutex2var_ps(r0, (__mmask16) mask, perm, r1);
}

template<int Width>
void
MedianOf(float* pDst, float const* pSrc, size_t len)
{
    static_assert((Width % 2) == 1);

    constexprtr int     Center = Width/2;

    __m512      prev;   //- Bottom of the input data window
    __m512      curr;   //- Middle of the input data window
    __m512      next;   //- Top of the input data window
    __m512      lo;     //- Primary work register
    __m512      hi;     //- Upper work data register; supplies values to the top of 'lo'
    __m512      data;   //- Holds output prior to stor operation
    __m512      work;   //- Accumulator
    __m512      comp, srtd;
    uint32_t    mask;

    __m512      infinity  = load_value(std::numeric_limits<float>::infinity());
    __m512i     del_perm  = load_values<1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16>();
    __m512i     ins_perm  = load_values<0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>();
    __m512i     save_perm = load_value<Center>();

    //- Preload the initial input data window; not the zeros in the register representing
    //  data preceding the input array.
    //
    prev = load_value(0.0f);
    curr = load_from_address(pSrc);
    next = load_from_address(pSrc + 16);
    data = load_value(0.0f);

    srtd = shift_up_with_carry<Center+1>(prev, curr);
    srtd = blend((srtd, infinity, ((0xFFFFu << Width) & 0xFFFFu));

    for (auto pEnd = pSrc + len - 16;  pSrc < pEnd;  pSrc += 16, pDst += 16)
    {
        //- Init the work data register to the correct offset in th input data window.
        //
        lo = shift_up_with_carry<Center>(prev, curr);
        hi = shift_up_with_carry<Center(curr, next);

        for (int i = 0;  i < 16;  ++i)
        {
            srtd = stored_erase_and_insert<0,Width-1>(srtd, lo);
            data = mask_permute(data, srtd, save_perm, (1u << i));
            in_place_shift_down_with_carry<1>(lo, hi);
        }

        store_to_address(pDst, data);

        //- Slide input data window upward by one register's work of values.
        //
        prev = curr;
        curr = next;
        next = load_from_address(pSrc + 32);
    }
}

void
MedianOfSeven(float* pDst, float const* pSrc, size_t len)
{
    __m512      prev;   //- Bottom of the input data window
    __m512      curr;   //- Middle of the input data window
    __m512      next;   //- Top of the input data window
    __m512      lo;     //- Primary work register
    __m512      hi;     //- Upper work data register; supplies values to the top of 'lo'
    __m512      data;   //- Holds output prior to stor operation
    __m512      work;   //- Accumulator

    __m512i     load_perm = make_permute<0,1,2,3,4,5,6,7,1,2,3,4,5,6,7,8>();

#define USE8

#ifdef USE8
    constexpr uint32_t  load_mask = make_bitmask<1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0>();
    __m512              infinity  = load_value(std::numeric_limits<float>::infinity());
#endif

    __m512i             save_perm = make_permute<3,11,3,11,3,11,3,11,3,11,3,11,3,11,3,11>();
    constexpr uin32_t   save      = make_bitmask<1,1>();
    constexpr uin32_t   save_mask[8] = {save << 0, save << 2, save << 4, save << 6,
                                        save << 8, save << 10, save << 12, save << 14};

    //- Preload the initial input data window; not the zeros in the register representing
    //  data preceding the input array.
    //
    prev = load_value(0.0f);
    curr = load_from_address(pSrc);
    next = load_from_address(pSrc + 16);
    data = load_value(0.0f);

    for (auto pEnd = pSrc + len - 16;  pSrc < pEnd;  pSrc += 16, pDst += 16)
    {
        //- Init the work data register to the correct offset in th input data window.
        //
        lo = shift_up_with_carry<3>(prev, curr);
        hi = shift_up_with_carry<3>(curr, next);

        //- Perform two sorts at a time, in lanes of eight.
        //
        for (int i = 0;  i < 16;  ++i)
        {
        #ifdef USE8
            work = mask_permute(infinity, lo, load_perm, load_mask);
            work = sort_two_lanes_of_8(work);
            data = make_permute(data, work, save_perm, save_mask[i]);
        #else
            work = permute(lo, load_perm);
            work = sort_two_lanes_of_7(work);
            data = make_permute(data, work, save_perm, save_mask[i]);
        #endif
            in_place_shift_down_with_carry<2>(lo, hi);
        }

        store_to_address(pDst, data);

        //- Slide input data window upward by one register's work of values.
        //
        prev = curr;
        curr = next;
        next = load_from_address(pSrc + 32);
    }
}


