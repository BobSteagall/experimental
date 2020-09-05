#include "test.hpp"
#include <algorithm>
#include <random>
#include <vector>

#include <errno.h>
#include <pthread.h>
#include <unistd.h>

using namespace simd;
using namespace std;

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"

namespace {
    size_t              random_count = (1 << 24);
    std::vector<float>  random_values;
    std::mt19937        rgen(19690720);

    bool const  do_print = false;
}

void
pin_thread()
{
    cpu_set_t   cpuset;
    pthread_t   thread = pthread_self();

    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);

    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0)
    {
        printf("couldn't set pthread_setaffinity_np\n");
    }
}

void
load_random_values()
{
    if (random_values.size() == 0)
    {
        std::uniform_int_distribution   dist(-50, 49);

        random_values.resize(random_count);

        for (auto& v : random_values)
        {
            v = dist(rgen);
        }
    }
}

void
tf01()
{
    ri512  r1 = load_values<1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16>();
//    PRINT_REG(r1);

//    PRINT_MASK(0xAA);

    fr512   r2 = load_values(16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f,
                              8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
//    PRINT_REG(r2);

    fr512   r3 = sort_two_lanes_of_8(r2);
//    PRINT_REG(r3);

    rf512   r4 = sort_two_lanes_of_7(r2);
//    PRINT_REG(r4);
}

void
tf02()
{
    int     N = -1;
    float   vals[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float   exp[16] = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8};
    float   work[16];
    rf512   rv, rr;

    do
    {
        for (int i = 0;  i < 8;  ++i)
        {
            work[i+8] = work[i] = vals[i];
        }

        rv = load_from(&work[0]);
        rr = sort_two_lanes_of_8(rv);

        store_to_address(work, rr);

        if (do_print  &&  (++N % 1000) == 0)
        {
            printf("N = %d\n", N);
            PRINT_REG(rv);
            PRINT_REG(rr);
            PRINT_LINE();
        }

        if (!std::equal(std::begin(exp), std::end(exp), std::begin(work)))
        {
            printf("error in combo\n");
            PRINT_REG(rv);
            PRINT_REG(rr);
        }
    }
    while (std::next_permutation(std::begin(vals), std::end(vals)));
}

void
tf03()
{
    int     N = -1;
    float   vals[7]  = {1, 2, 3, 4, 5, 6, 7};
    float   exp[16]  = {1, 2, 3, 4, 5, 6, 7, 99, 1, 2, 3, 4, 5, 6, 7, 99};
    float   work[16] = {0, 0, 0, 0, 0, 0, 0, 99, 0, 0, 0, 0, 0, 0, 0, 99};
    rf512   rv, rr;

    do
    {
        for (int i = 0;  i < 7;  ++i)
        {
            work[i+8] = work[i] = vals[i];
        }

        rv = load_from(work);
        rr = sort_two_lanes_of_8(rv);

        store_to_address(work, rr);

        if (do_print  &&  (++N % 100) == 0)
        {
            printf("N = %d\n", N);
            PRINT_REG(rv);
            PRINT_REG(rr);
            PRINT_LINE();
        }

        if (!std::equal(std::begin(exp), std::end(exp), std::begin(work)))
        {
            printf("error in combo\n");
            PRINT_REG(rv);
            PRINT_REG(rr);
        }
    }
    while (std::next_permutation(std::begin(vals), std::end(vals)));
}

void
simple_median_of_7(std::vector<float>& vdst, std::vector<float> const& vsrc)
{
    float   tmp[7];
    size_t  ilast = vsrc.size();
    auto    tmp_f = std::begin(tmp);
    auto    tmp_l = std::end(tmp);
    auto    src_f = std::begin(vsrc);
    auto    src_l = std::end(vsrc);


    std::fill(tmp_f, tmp_f+3, vsrc.front());
    std::copy(src_f, src_f+4, tmp_f+3);
    std::sort(tmp_f, tmp_l);
    vdst[0] = tmp[3];

    std::fill(tmp_f, tmp_f+2, vsrc.front());
    std::copy(src_f, src_f+5, tmp_f+2);
    std::sort(tmp_f, tmp_l);
    vdst[1] = tmp[3];

    std::fill(tmp_f, tmp_f+1, vsrc.front());
    std::copy(src_f, src_f+6, tmp_f+1);
    std::sort(tmp_f, tmp_l);
    vdst[2] = tmp[3];

    for (size_t i = 3;  i < vsrc.size()-3;  ++i, ++src_f)
    {
        std::copy(src_f, src_f + 7, tmp_f);
        std::sort(tmp_f, tmp_l);
        vdst[i] = tmp[3];
    }

    std::fill(tmp_f, tmp_l, vsrc.back());
    std::copy(src_l - 6, src_l, tmp_f);
    std::sort(tmp_f, tmp_l);
    vdst[ilast - 3] = tmp[3];

    std::fill(tmp_f, tmp_l, vsrc.back());
    std::copy(src_l - 5, src_l, tmp_f);
    std::sort(tmp_f, tmp_l);
    vdst[ilast - 2] = tmp[3];

    std::fill(tmp_f, tmp_l, vsrc.back());
    std::copy(src_l - 4, src_l, tmp_f);
    std::sort(tmp_f, tmp_l);
    vdst[ilast - 1] = tmp[3];

}


void
x_median_of_7(float* pdst, float const* psrc, size_t const buf_len)
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


void
tf04()
{
    std::vector<float>  vsrc, vdst_avx, vdst_stl;
    size_t              ncnt = 8;

    for (int i = 0;  i < 50;  ++i, ++ncnt)
    {
        vsrc.resize(ncnt);
        vdst_stl.resize(ncnt+1);
        vdst_avx.resize(ncnt+1);

        std::fill(std::begin(vdst_stl), std::end(vdst_stl), 99.0f);
        std::fill(std::begin(vdst_avx), std::end(vdst_avx), 99.0f);

        for (size_t i = 0;  i < vsrc.size();  ++i)
        {
            vsrc[i] = i;
        }

        simple_median_of_7(vdst_stl, vsrc);
        x_median_of_7(vdst_avx.data(), vsrc.data(), vsrc.size());

        if (vdst_avx.back() != 99.0f)
        {
            printf("buffer overrun at size %ld\n", ncnt);
        }

        for (size_t i = 0;  i < vsrc.size();  ++i)
        {
            if (vdst_avx[i] != vdst_stl[i])
            {
                printf("size %ld: diff at index %ld: avx = %.1f  stl = %.1f\n",
                        ncnt, i, vdst_avx[i], vdst_stl[i]);
            }
        }
    }
}

void
tf05()
{
//    return;
    load_random_values();

    std::vector<float>  vsrc, vdst_avx, vdst_stl;
    stopwatch           sw;
    size_t const        ncnt = 1048576;
    int const           reps = 10;

    vsrc.resize(ncnt);
    vdst_avx.resize(ncnt);
    vdst_stl.resize(ncnt);

    for (size_t i = 0;  i < vsrc.size();  ++i)
    {
        vsrc[i] = random_values[i];
    }

    sw.start();
    for (int i = 0;  i < reps;  ++i)
    {
        simple_median_of_7(vdst_stl, vsrc);
    }
    sw.stop();
    printf("simple median (random): %ld usec\n", sw.microseconds_elapsed()/reps);

    sw.start();
    for (int i = 0;  i < reps;  ++i)
    {
        x_median_of_7(vdst_avx.data(), vsrc.data(), vsrc.size());
    }
    sw.stop();
    printf("avx512 median (random): %ld usec\n", sw.microseconds_elapsed()/reps);


    for (size_t i = 0;  i < vsrc.size();  ++i)
    {
        vsrc[i] = i;
    }

    sw.start();
    for (int i = 0;  i < reps;  ++i)
    {
        simple_median_of_7(vdst_stl, vsrc);
    }
    sw.stop();
    printf("simple median (sorted): %ld usec\n", sw.microseconds_elapsed()/reps);

    sw.start();
    for (int i = 0;  i < reps;  ++i)
    {
        x_median_of_7(vdst_avx.data(), vsrc.data(), vsrc.size());
    }
    sw.stop();
    printf("avx512 median (sorted): %ld usec\n", sw.microseconds_elapsed()/reps);

    for (size_t i = 0;  i < vsrc.size();  ++i)
    {
        if (vdst_avx[i] != vdst_stl[i])
        {
            printf("median diff at index %ld: vdst_avx = %.1f  vdst_stl = %.1f\n", i, vdst_avx[i], vdst_stl[i]);
        }
    }
}
