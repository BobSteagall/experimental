#ifdef __OPTIMIZE__
    #include <immintrin.h>
    #define FORCE_INLINE    __attribute__((__always_inline__)) inline
#else
    #define __OPTIMIZE__
    #include <immintrin.h>
    #undef __OPTIMIZE__
    #define FORCE_INLINE    inline
#endif

using rd_128 = rd_128;
using rf_128 = rf_128;
using ri_128 = ri_128;

using rd_256  = rd_256;
using rf_256  = rf_256;
using ri_256  = ri_256;
using msk_256 = ri_256;

using rd_512  = rd_512;
using rf_512  = rf_512;
using ri_512  = ri_512;
using msk_512 = uint32_t;

void print(char const* regname, rf_256 const& reg)
{
    float   values[8];

    __m256_storeu_ps(values, reg);

    //- check pointer

    for (i= 0; i < 8)
    //- print values
}

void printm(char const* maskname, uint32_t mask, size_t)

void printm(char const* maskname, ri_256 mask, size_t)
{
    int32_t     values[8];
    __mm256_maskstore_epi32(values, _mm256_set1_epi32(-1), mask);
    //print values[]
}


load_from_address(float const* p)

template<int I> rf_512
load_from_element(rf_512 r0)
{
    static_assert(I >= 0 && I < 16)
    return _mm512_permutexvar_ps(_mm512_set_epi32(I), r0);
}

rf_512 load_lower_element(rf_512 r0); _mm512_permutexvar_ps(_mm512_set1_epi32(0), r0);
rf_512 load_upper_element(rf_512 r0);

rf_512(i/d) load_value(float/int32_t/double)

template<int A, ...., int P> ri_512 load_values();

void store_to_address(void*, rf_512)

rf_512
blend(rf_512 r0, rf_512 r1, uint32_t mask)
{
    return _mm512_mask_blend_ps((__mmask16) mask, r0, r1);
}


template<int A, ..., int P> rf_512
permute(rf_512 r0)
{
    return _mm512_permutexvar_ps(_mm512_setr_epi32(A,...,P), r0);
}

template<int R> rf_512
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
        constexpr int    B = (S + 2) % 16;
        ...
        constexpr int    P = (S + 16) % 16;

        return _mm512_permutexvar_ps(_mm512_setr_epi32(A,...,P), r0);
    }
}

template<int R> rf_512
rotate_down(rf_512 r0)
{
    static_assert(R >= 0);
    return rotate<-R>(r0);
}

template<int R> rf_512
rotate_up(rf_512 r0)
{
    static_assert(R >= 0);
    return rotate<R>(r0);
}

template<int KernelSize, int KernelCenter>
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

    rf_512  prev;   //- Bottom of the input data window
    rf_512  curr;   //- Middle of the input data windows
    rf_512  next;   //- Top of the input data window
    rf_512  lo;     //- Primary work data register, used to multiply kernel coefficients
    rf_512  hi;     //- Upper work data register, supplies values to the top of 'lo'
    rf_512  sum;    //- Accumulated values register

    rf_512  kcoeff[KernelSize];     //- Coefficients fot eh convolution kernel

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

template<int S> rf_512
shift_up(rf_512 r0)
{
    return blend(rotate_up<S>(r0), load_value(0), shift_up_blend_mask<S>());
}

template<int S> rf_512
shift_down(rf_512 r0)
{
    return blend(rotate_down<S>(r0), load_value(0), shift_down_blend_mask<S>());
}

template<int S> rf_512
shift_up_with_carry(rf_512 lo, rf_512 hi)
{
    return blend(rotate_up<S>(lo), rotate_up<S>(hi), shift_up_blend_mask<S>());
}

template<int S> rf_512
shift_down_with_carry(rf_512 lo, rf_512 hi)
{
    return blend(rotate_down<S>(lo), rotate_down<S>(hi), shift_down_blend_mask<S>());
}

template<int S> rf_512
shift_up_and_fill(rf_512 r0, float fill)
{
    return blend(rotate_up<S>(r0), load_value(fill), shift_up_blend_mask<S>());
}

template<int S> rf_512
shift_down_and_fill(rf_512 r0, float fill)
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
in_place_shift_down_with_carry(rf_512& lo, rf_512& hi)
{
    static_assert(S >= 0  &&  S <= 16);

    constexpr uint32_t  zmask = (0xFFFFu >> (unsigned) S);
    constexpr uint32_t  bmask = ~zmask & 0xFFFFu;
    ri_512             perm  = make_shift_permutation<S, bmask>();

    lo = _mm512_permutex2var_ps(lo, perm, hi);
    hi = _mm512_maskz_permutex2var_ps((__mask16) zmask, hi, perm, hi);
}

template<int BIAS, uint32_t MASK> ri_512
make_shift_permutation()
{
    constexpr int32_t   a = ((BIAS + 0)  % 16) | ((MASK 1u)        ? 0x10 : 0);
    constexpr int32_t   b = ((BIAS + 1)  % 16) | ((MASK 1u << 1u)  ? 0x10 : 0);
    constexpr int32_t   c = ((BIAS + 1)  % 16) | ((MASK 1u << 2u)  ? 0x10 : 0);
    ...
    constexpr int32_t   p = ((BIAS + 15) % 16) | ((MASK 1u << 15u) ? 0x10 : 0);

    return _mm512_setr_epi32(a,...,p);
}


rf_512
compare_with_exchange(rf_512 vals, ri_512 perm, uint32_t mask)
{
    rf_512  exch = permute(vals, perm);
    rf_512  vmin = minimum(vals, exch);
    rf_512  vmax = maximum(vals, exch);

    return blend(vmin, vmax, mask);
}

rf_512
minimum(rf_512 r0, rf_512 r1)
{
    return _mm512_min_ps(r0, r1);
}

rf_512
maximum(rf_512 r0, rf_512 r1)
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

template<unsigned A, ..., unsigned P> ri_512
make_permute()
{
    static_assert((A < 16) && ... && (P < 16));
    return _mm512_setr_epi32(A,...,P):
}


rf_512
sort_two_lanes_of_8(rf_512 vals)
{
    //- Precompute the permutations and bitmasks for the siz stages of this bitonic sorting sequence.
    //                                       0  1  2  3  4  5  6  6   0  1  2  3  4  5  6  7
    //                                       ---------------------------------------------------
    ri_512 const       perm0 = make_permute<4, 5, 6, 3, 0, 1, 2, 7, 12,13,14,11, 8, 9,10,15>();
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
unpack_complex(rf_512& real, rf_512& imag, rf_512 cxlo, rf_512 cxhi)
{
    ri_512     lo_perm = _mm512_setr_epi32(0,2,4,6,8,10,12.14,1,3,5,7,9,11,13,15);
    ri_512     hi_perm = _mm512_setr_epi32(1,3,5,7,9,11,13,15,0,2,4,6,8,10,12,14);
    ri_512     sw_perm = _mm256_set1_epi32(8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7);

    cxlo = _mm512_permutexvar_ps(lo_perm, cxlo);
    cxhi = _mm512_permutexvar_ps(hi_perm, cxhi);

    real = _mm512_mask_blend_ps(0b1111'1111'0000'0000, cxlo, cxhi);
    imag = _mm512_mask_blend_ps(0b0000'0000'1111'1111, cxlo, cxhi);
    imag = _mm512_permutexvar_ps(sw_perm, imag);
}

rf_512
shift_up_1(rf_512 r0)
{
    ri_512             perm = _mm512_setr_epi32(15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14);
    constexpr __mmask16 mask = 0b1111'1111'1111'1110;

    return _mm512_maskz_permutexvar_ps(mask, perm, r0);
}

template<int E, int I>
sorted_erase_and_insert(rf_512 srtd, rf_512 data)
{
    rf_512      infinity = load_value(std::numeric_limits<float>::infinity());
    ri_512     del_perm = load_values<1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16>();
    ri_512     ins_perm = load_values<0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>();

    rf_512      work, comp;
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

rf_512
mask_permute(rf_512 r0, rf_512 r1, ri_512 perm, uint32_t mask)
{
    return _mm512_mask_permutexvar_ps(r0, (__mmask16) mask, perm, r1);
}

rf_512
mask_permute2(rf_512 r0, rf_512 r1, ri_512 perm, uint32_t mask)
{
    return _mm512_mask_permutex2var_ps(r0, (__mmask16) mask, perm, r1);
}

rf_512
sum_of_squares(rf_512 r0, rf_512 r1)
{
    return _mm512_add_ps(_mm512_mul_ps(r0, r0), _mm512_mul_ps(r1, r1));
}

rf_512
square_root(rf_512 r0)
{
    return _mm512_sqrt_ps(r0);
}

void
magn(float* pdst, cxfloat const* psrc, size_t count)
{
    rf_512  lo, hi, real, imag, norm, magn;

    for (auto pend = psrc + count;  psrc < pend;  psrc += 16, pdst += 16)
    {
        lo = load_from_address(psrc);
        lo = permute<0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15>(lo);

        hi = load_from_address(psrc + 8);
        hi = permute<1,3,5,7,9,11,13,15,0,2,4,6,8,10,12,14>(hi);

        real = blend(lo, hi, 0b1111'1111'0000'0000);
        imag = blend(lo, hi, 0b0000'0000'1111'1111);
        imag = permute<8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7>(imag);

        norm = sum_of_squares(real, imag);
        magn = square_root(norm);
        store_to_address(pdst, magn);
    }
}

template<int Width>
void
MedianOf(float* pDst, float const* pSrc, size_t len)
{
    static_assert((Width % 2) == 1);

    constexprtr int     Center = Width/2;

    rf_512      prev;   //- Bottom of the input data window
    rf_512      curr;   //- Middle of the input data window
    rf_512      next;   //- Top of the input data window
    rf_512      lo;     //- Primary work register
    rf_512      hi;     //- Upper work data register; supplies values to the top of 'lo'
    rf_512      data;   //- Holds output prior to stor operation
    rf_512      work;   //- Accumulator
    rf_512      comp, srtd;
    uint32_t    mask;

    rf_512      infinity  = load_value(std::numeric_limits<float>::infinity());
    ri_512     del_perm  = load_values<1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16>();
    ri_512     ins_perm  = load_values<0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14>();
    ri_512     save_perm = load_value<Center>();

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
    rf_512      prev;   //- Bottom of the input data window
    rf_512      curr;   //- Middle of the input data window
    rf_512      next;   //- Top of the input data window
    rf_512      lo;     //- Primary work register
    rf_512      hi;     //- Upper work data register; supplies values to the top of 'lo'
    rf_512      data;   //- Holds output prior to stor operation
    rf_512      work;   //- Accumulator

    ri_512     load_perm = make_permute<0,1,2,3,4,5,6,7,1,2,3,4,5,6,7,8>();

#define USE8

#ifdef USE8
    constexpr uint32_t  load_mask = make_bitmask<1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0>();
    rf_512              infinity  = load_value(std::numeric_limits<float>::infinity());
#endif

    ri_512             save_perm = make_permute<3,11,3,11,3,11,3,11,3,11,3,11,3,11,3,11>();
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


