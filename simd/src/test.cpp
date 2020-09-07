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

    bool const  do_print  = false;
    int const   time_reps = 100;
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

    __m128i     chunk;

    chunk = _mm_setr_epi32(1, 2, 3, 4);
    PRINT_REG(chunk);
    chunk = _mm_shuffle_epi32(chunk, _MM_SHUFFLE(1, 0, 3, 2));
    PRINT_REG(chunk);

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
stl_median_of_7(std::vector<float>& vdst, std::vector<float> const& vsrc)
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
tf04()
{
    std::vector<float>  vsrc, vdst_avx, vdst_stl;
    size_t              ncnt = 7;

    for (int i = 0;  i < 121;  ++i, ++ncnt)
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

        stl_median_of_7(vdst_stl, vsrc);
        avx_median_of_7(vdst_avx.data(), vsrc.data(), vsrc.size());

        if (vdst_stl.back() != 99.0f)
        {
            printf("stl buffer overrun at size %ld\n", ncnt);
        }

        if (vdst_avx.back() != 99.0f)
        {
            printf("avx buffer overrun at size %ld\n", ncnt);
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
median_rep_driver(std::vector<float> const& vsrc, size_t reps, char const* name)
{
    std::vector<float>  vdst_avx, vdst_stl;     //- destination arrays
    stopwatch           sw;                     //- for timing
    size_t const        ncnt = vsrc.size();     //- number of elements in source
    size_t const        xmin = 100'000'000u;    //- min value of (reps * ncnt)
    int64_t             npr;                    //- avg nanosecs per repetition
    double              npe;                    //- avg nanosecs per element

    //- Try to do enough reps for a reasonable amount of collection time.
    //
    if ((reps * ncnt) < xmin)
    {
        reps = xmin / ncnt;
    }

    //- Resize the destination arrays, make sure there's an extra element at the end of the
    //  arrays to test for overruns.  Fill with flag values.
    //
    vdst_avx.resize(ncnt+1);
    vdst_stl.resize(ncnt+1);
    std::fill(std::begin(vdst_stl), std::end(vdst_stl), 99.0f);
    std::fill(std::begin(vdst_avx), std::end(vdst_avx), 99.0f);

    //- Compute the median using the simple STL-based algorithm.
    //
    sw.start();
    for (size_t i = 0;  i < reps;  ++i)
    {
        stl_median_of_7(vdst_stl, vsrc);
    }
    sw.stop();
    npr = sw.nanoseconds_elapsed()/reps;
    npe = (double) npr / (double) ncnt;
    printf("stl %s %8ld %9ld %8.3f\n", name, ncnt, npr, npe);

    //- Compute the median using the AVX512-based algorithm.
    //
    sw.start();
    for (size_t i = 0;  i < reps;  ++i)
    {
        avx_median_of_7(vdst_avx.data(), vsrc.data(), vsrc.size());
    }
    sw.stop();
    npr = sw.nanoseconds_elapsed()/reps;
    npe = (double) npr / (double) ncnt;
    printf("avx %s %8ld %9ld %8.3f\n", name, ncnt, npr, npe);

    //- Check for overruns.
    //
    if (vdst_stl.back() != 99.0f)
    {
        printf("stl buffer overrun at size %ld\n", ncnt);
    }
    if (vdst_avx.back() != 99.0f)
    {
        printf("avx buffer overrun at size %ld\n", ncnt);
    }

    //- Verify that the two algorithms give identical results.
    //
    for (size_t i = 0;  i < ncnt;  ++i)
    {
        if (vdst_avx[i] != vdst_stl[i])
        {
            printf("(%s)) diff at index %ld: vdst_avx = %.1f  vdst_stl = %.1f\n",
                   name, i, vdst_avx[i], vdst_stl[i]);
        }
    }
}

void
tf05()
{
    load_random_values();

    std::vector<float>  vsrc;
    size_t const        min_ncnt = 100u;
    size_t const        max_ncnt = 10'000'000u;
    int const           tmg_reps = 100;

    for (size_t ncnt = min_ncnt;  ncnt <= max_ncnt;  ncnt *= 10)
    {
        vsrc.resize(ncnt);

        for (size_t i = 0;  i < vsrc.size();  ++i)
        {
            vsrc[i] = i;
        }
        median_rep_driver(vsrc, tmg_reps, "sorted");
    }

    for (size_t ncnt = min_ncnt;  ncnt <= max_ncnt;  ncnt *= 10)
    {
        vsrc.resize(ncnt);

        for (size_t i = 0;  i < vsrc.size();  ++i)
        {
            vsrc[i] = random_values[i];
        }
        median_rep_driver(vsrc, tmg_reps, "random");
    }
}

/*
void
tf06()
{
    load_random_values();

    std::vector<float>  vsrc, vdst_avx, vdst_stl;
    stopwatch           sw;
    size_t const        ncnt = 1048576;
    int const           reps = 100;

    vsrc.resize(ncnt);
    vdst_avx.resize(ncnt+1);
    vdst_stl.resize(ncnt+1);

    for (size_t i = 0;  i < vsrc.size();  ++i)
    {
        vsrc[i] = i;
    }
    std::fill(std::begin(vdst_stl), std::end(vdst_stl), 99.0f);
    std::fill(std::begin(vdst_avx), std::end(vdst_avx), 99.0f);

    sw.start();
    for (int i = 0;  i < reps;  ++i)
    {
        stl_median_of_7(vdst_stl, vsrc);
    }
    sw.stop();
    printf("simple median (sorted): %ld usec\n", sw.microseconds_elapsed()/reps);

    sw.start();
    for (int i = 0;  i < reps;  ++i)
    {
        avx_median_of_7(vdst_avx.data(), vsrc.data(), vsrc.size());
    }
    sw.stop();
    printf("avx512 median (sorted): %ld usec\n", sw.microseconds_elapsed()/reps);

    for (size_t i = 0;  i < vsrc.size();  ++i)
    {
        vsrc[i] = random_values[i];
    }
    std::fill(std::begin(vdst_stl), std::end(vdst_stl), 99.0f);
    std::fill(std::begin(vdst_avx), std::end(vdst_avx), 99.0f);

    sw.start();
    for (int i = 0;  i < reps;  ++i)
    {
        stl_median_of_7(vdst_stl, vsrc);
    }
    sw.stop();
    printf("simple median (random): %ld usec\n", sw.microseconds_elapsed()/reps);

    sw.start();
    for (int i = 0;  i < reps;  ++i)
    {
        avx_median_of_7(vdst_avx.data(), vsrc.data(), vsrc.size());
    }
    sw.stop();
    printf("avx512 median (random): %ld usec\n", sw.microseconds_elapsed()/reps);

    if (vdst_stl.back() != 99.0f)
    {
        printf("stl buffer overrun at size %ld\n", ncnt);
    }

    if (vdst_avx.back() != 99.0f)
    {
        printf("avx buffer overrun at size %ld\n", ncnt);
    }

    for (size_t i = 0;  i < vsrc.size();  ++i)
    {
        if (vdst_avx[i] != vdst_stl[i])
        {
            printf("median diff at index %ld: vdst_avx = %.1f  vdst_stl = %.1f\n", i, vdst_avx[i], vdst_stl[i]);
        }
    }
}
*/
