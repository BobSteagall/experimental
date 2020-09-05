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
print_reg(char const* pname, rd256 r)
{
    double  vals[4];

    _mm256_storeu_pd(&vals[0], r);
    print_vals(pname, vals, 4);
}

void
print_reg(char const* pname, rf256 r)
{
    float   vals[8];

    _mm256_storeu_ps(&vals[0], r);
    print_vals(pname, vals, 8);
}

void
print_reg(char const* pname, ri256 r)
{
    int32_t vals[8];

    _mm256_storeu_si256((ri256*) &vals[0], r);
    print_vals(pname, vals, 8);
}

//------
//
void
print_reg(char const* pname, rd512 r)
{
    double  vals[8];

    _mm512_storeu_pd(&vals[0], r);
    print_vals(pname, vals, 8);
}

void
print_reg(char const* pname, rf512 r)
{
    float   vals[16];

    _mm512_storeu_ps(&vals[0], r);
    print_vals(pname, vals, 16);
}

void
print_reg(char const* pname, ri512 r)
{
    int32_t vals[16];

    _mm512_storeu_epi32(&vals[0], r);
    print_vals(pname, vals, 16);
}

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
print_mask(char const* pname, __m256i mask, int)
{
    if (pname != nullptr)
    {
        printf("mask %s:\n", pname);
    }
    print_reg(nullptr, mask);
}


void
median_of_7(float* pdst, float const* psrc, size_t const buf_len)
{
    __m512      prev;   //- Bottom of the input data window
    __m512      curr;   //- Middle of the input data window
    __m512      next;   //- Top of the input data window
    __m512      lo;     //- Primary work register
    __m512      hi;     //- Upper work data register; feeds values into the top of 'lo'
    __m512      data;   //- Holds output prior to store operation
    __m512      work;   //- Accumulator
    m512        mask;   //- Trailing boundary mask

    rf512 const     first = load_value(psrc[0]);
    rf512 const     last  = load_value(psrc[buf_len - 1]);

    //- This permutation specifies how to load the two lanes of 7.
    //
    ri512 const     load_perm = make_permute<0,1,2,3,4,5,6,7,1,2,3,4,5,6,7,8>();

    //- This permutation specifies which elements to save.
    //
    ri512 const     save_perm = make_permute<3,11,3,11,3,11,3,11,3,11,3,11,3,11,3,11>();

    //- This is a bitmask pattern for picking out adjacent elements.
    //
    constexpr m512  save = make_bitmask<1,1>();

    //- This array of bitmasks specifies which pair of elements to blend into the result.
    //
    constexpr m512  save_mask[8] = {save << 0, save << 2,  save << 4,  save << 6,
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
            data = mask_permute(data, work, save_perm, save_mask[i]);
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
                data = mask_permute(data, work, save_perm, save_mask[i]);
                in_place_shift_down_with_carry<2>(lo, hi);
            }

            if (wrote <= (buf_len - 16))
            {
                store_to_address(pdst + wrote, data);
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
