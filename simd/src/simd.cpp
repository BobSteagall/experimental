#include "simd.hpp"

namespace simd {

void
print_vals(char const* pname, double const* pvals, int n)
{
    if (pname != nullptr)
    {
        printf("reg %s:\n", pname);
    }

    for (int i = 0;  i < n;  ++i)
    {
        printf("%6.1f", pvals[i]);
    }
    printf("\n");
    fflush(stdout);
}

void
print_vals(char const* pname, float const* pvals, int n)
{
    if (pname != nullptr)
    {
        printf("reg %s:\n", pname);
    }

    for (int i = 0;  i < n;  ++i)
    {
        printf("%6.1f", pvals[i]);
    }
    printf("\n");
    fflush(stdout);
}

void
print_vals(char const* pname, int32_t const* pvals, int n)
{
    if (pname != nullptr)
    {
        printf("reg %s:\n", pname);
    }

    for (int i = 0;  i < n;  ++i)
    {
        printf("%6d", pvals[i]);
    }
    printf("\n");
    fflush(stdout);
}


void
print_reg(char const* pname, uint32_t i);

//------
//
void
print_reg(char const* pname, rd_128 r)
{
    double  vals[2];

    _mm_storeu_pd(&vals[0], r);
    print_vals(pname, vals, 2);
}

void
print_reg(char const* pname, rf_128 r)
{
    float   vals[4];

    _mm_storeu_ps(&vals[0], r);
    print_vals(pname, vals, 4);
}

void
print_reg(char const* pname, ri_128 r)
{
    int32_t vals[4];

    _mm_storeu_epi32(&vals[0], r);
    print_vals(pname, vals, 4);
}

//------
//
void
print_reg(char const* pname, rd_256 r)
{
    double  vals[4];

    _mm256_storeu_pd(&vals[0], r);
    print_vals(pname, vals, 4);
}

void
print_reg(char const* pname, rf_256 r)
{
    float   vals[8];

    _mm256_storeu_ps(&vals[0], r);
    print_vals(pname, vals, 8);
}

void
print_reg(char const* pname, ri_256 r)
{
    int32_t vals[8];

    _mm256_storeu_si256((ri_256*) &vals[0], r);
    print_vals(pname, vals, 8);
}

//------
//
void
print_reg(char const* pname, rd_512 r)
{
    double  vals[8];

    _mm512_storeu_pd(&vals[0], r);
    print_vals(pname, vals, 8);
}

void
print_reg(char const* pname, rf_512 r)
{
    float   vals[16];

    _mm512_storeu_ps(&vals[0], r);
    print_vals(pname, vals, 16);
}

void
print_reg(char const* pname, ri_512 r)
{
    int32_t vals[16];

    _mm512_storeu_epi32(&vals[0], r);
    print_vals(pname, vals, 16);
}

//------
//
void
print_mask(char const* pname, uint32_t mask, int bits)
{
    if (pname != nullptr)
    {
        printf("mask %s:\n", pname);
    }

    uint32_t    probe = 1;

    for (int i = 0;  i < bits;  ++i, probe <<= 1)
    {
        printf("%6d", (mask & probe) ? 1 : 0);
    }
    printf("\n");
    fflush(stdout);
}

void
print_mask(char const* pname, ri_256 mask, int)
{
    if (pname != nullptr)
    {
        printf("mask %s:\n", pname);
    }
    print_reg(nullptr, mask);
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

    rf_512  prev;   //- Bottom of the input data window
    rf_512  curr;   //- Middle of the input data windows
    rf_512  next;   //- Top of the input data window
    rf_512  lo;     //- Primary work data register, used to multiply kernel coefficients
    rf_512  hi;     //- Upper work data register, supplies values to the top of 'lo'
    rf_512  sum;    //- Accumulated value

    rf_512  kcoeff[KernelSize];     //- Coefficients of the convolution kernel

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

void
avx_symm_convolve(float* pdst, float const* pkrnl, size_t klen, float const* psrc, size_t len)
{
    switch (klen)
    {
      case 3:
        avx_convolve<3,1>(pdst, pkrnl, psrc, len);
        return;

      case 5:
        avx_convolve<5,2>(pdst, pkrnl, psrc, len);
        return;

      case 7:
        avx_convolve<7,3>(pdst, pkrnl, psrc, len);
        return;

      case 9:
        avx_convolve<9,4>(pdst, pkrnl, psrc, len);
        return;

      case 11:
        avx_convolve<11,5>(pdst, pkrnl, psrc, len);
        return;

      case 13:
        avx_convolve<13,6>(pdst, pkrnl, psrc, len);
        return;

      case 15:
        avx_convolve<15,7>(pdst, pkrnl, psrc, len);
        return;

      default:
        return;
    }
}

void
avx_median_of_7(float* pdst, float const* psrc, size_t const buf_len)
{
    rf_512      prev;   //- Bottom of the input data window
    rf_512      curr;   //- Middle of the input data window
    rf_512      next;   //- Top of the input data window
    rf_512      lo;     //- Primary work register
    rf_512      hi;     //- Upper work data register; feeds values into the top of 'lo'
    rf_512      data;   //- Holds output prior to store operation
    rf_512      work;   //- Accumulator
    msk_512        mask;   //- Trailing boundary mask

    rf_512 const     first = load_value(psrc[0]);
    rf_512 const     last  = load_value(psrc[buf_len - 1]);

    //- This permutation specifies how to load the two lanes of 7.
    //
    ri_512 const     load_perm = make_perm_map<0,1,2,3,4,5,6,7,1,2,3,4,5,6,7,8>();

    //- This permutation specifies which elements to save.
    //
    ri_512 const     save_perm = make_perm_map<3,11,3,11,3,11,3,11,3,11,3,11,3,11,3,11>();

    //- This is a bitmask pattern for picking out adjacent elements.
    //
    constexpr msk_512  save = make_bit_mask<1,1>();

    //- This array of bitmasks specifies which pair of elements to blend into the result.
    //
    constexpr msk_512  save_mask[8] = {save << 0, save << 2,  save << 4,  save << 6,
                                    save << 8, save << 10, save << 12, save << 14};

    //- Preload the initial input data window; note the values in the register representing
    //  data preceding the input array are equal to the first element.
    //

    if (buf_len < 16)
    {
        prev = first;
        mask = ~(0xffffffff << buf_len);
        curr = masked_load_from(psrc, last, mask);
        next = last;

        //- Init the work data register to the correct offset in the input data window.
        //
        lo = shift_up_with_carry<3>(prev, curr);
        hi = shift_up_with_carry<3>(curr, next);

        //- Perform two sorts at a time, in lanes of eight.
        //
        for (int i = 0;  i < 8;  ++i)
        {
            work = permute(lo, load_perm);
            work = sort_two_lanes_of_7(work);
            data = masked_permute(data, work, save_perm, save_mask[i]);
            in_place_shift_down_with_carry<2>(lo, hi);
        }

        masked_store_to(pdst, data, mask);
    }
    else
    {
        size_t  read  = 0;
        size_t  used  = 0;
        size_t  wrote = 0;

        curr  = first;
        next  = load_from(psrc);
        read += 16;
        used += 16;

        while (used < (buf_len + 16))
        {
            prev = curr;
            curr = next;

            if (read <= (buf_len - 16))
            {
                next  = load_from(psrc + read);
                read += 16;
            }
            else
            {
                mask = ~(0xffffffff << (buf_len - read));
                next = masked_load_from(psrc + read, last, mask);
                read = buf_len;
            }
            used += 16;

            //- Init the work data register to the correct offset in the input data window.
            //
            lo = shift_up_with_carry<3>(prev, curr);
            hi = shift_up_with_carry<3>(curr, next);

            //- Perform two sorts at a time, in lanes of eight.
            //
            for (int i = 0;  i < 8;  ++i)
            {
                work = permute(lo, load_perm);
                work = sort_two_lanes_of_7(work);
                data = masked_permute(data, work, save_perm, save_mask[i]);
                in_place_shift_down_with_carry<2>(lo, hi);
            }

            if (wrote <= (buf_len - 16))
            {
                store_to(pdst + wrote, data);
                wrote += 16;
            }
            else
            {
                mask = ~(0xffffffff << (buf_len - wrote));
                masked_store_to(pdst + wrote, data, mask);
                wrote = buf_len;
            }
        }
    }
}

}   //- simd namespace
