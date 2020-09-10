#include "test.hpp"
#include <algorithm>
#include <random>
#include <vector>

#include <errno.h>
#include <pthread.h>
#include <unistd.h>

#include <mkl.h>

#define MKL_Complex8    std::complex<float>

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
median_rep_driver
(vector<float> const& vsrc, size_t reps, char const* name, vector<string>& results)
{
    vector<float>   vdst_avx, vdst_stl;     //- destination arrays
    stopwatch       sw;                     //- for timing
    size_t const    ncnt = vsrc.size();     //- number of elements in source
    size_t const    xmin = 100'000'000u;    //- min value of (reps * ncnt)
    int64_t         stl_time;               //- avg nanosecs per rep for STL approach
    int64_t         avx_time;               //- avg nanosecs per rep for AVX approach
    double          npe;                    //- avg nanosecs per element
    char            resbuf[256];

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
    fill(begin(vdst_stl), end(vdst_stl), 99.0f);
    fill(begin(vdst_avx), end(vdst_avx), 99.0f);

    //- Compute the median using the simple STL-based algorithm.
    //
    sw.start();
    for (size_t i = 0;  i < reps;  ++i)
    {
        stl_median_of_7(vdst_stl, vsrc);
    }
    sw.stop();
    stl_time = sw.nanoseconds_elapsed()/reps;
    npe       = (double) stl_time / (double) ncnt;
    printf("stl %s %8ld %9ld %8.3f\n", name, ncnt, stl_time, npe);

    //- Compute the median using the AVX512-based algorithm.
    //
    sw.start();
    for (size_t i = 0;  i < reps;  ++i)
    {
        avx_median_of_7(vdst_avx.data(), vsrc.data(), vsrc.size());
    }
    sw.stop();
    avx_time = sw.nanoseconds_elapsed()/reps;
    npe      = (double) avx_time / (double) ncnt;
    printf("avx %s %8ld %9ld %8.3f\n", name, ncnt, avx_time, npe);

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
    fflush(stdout);

    double  speedup = (double) stl_time / (double) avx_time;
    sprintf(resbuf, "%s, %lu, %ld, %ld, %.2f, %lu",
            name, vsrc.size(), stl_time, avx_time, speedup, reps);
    results.push_back(string(resbuf));
}

void
tf05()
{
    load_random_values();

    vector<float>   vsrc;
    vector<string>  results;
    size_t const    min_ncnt = 100u;
    size_t const    max_ncnt = 10'000'000u;
    int const       tmg_reps = 100;

    for (size_t ncnt = min_ncnt;  ncnt <= max_ncnt;  ncnt *= 10)
    {
        vsrc.resize(ncnt);

        for (size_t i = 0;  i < vsrc.size();  ++i)
        {
            vsrc[i] = i;
        }
        median_rep_driver(vsrc, tmg_reps, "sorted", results);
    }

    for (size_t ncnt = min_ncnt;  ncnt <= max_ncnt;  ncnt *= 10)
    {
        vsrc.resize(ncnt);

        for (size_t i = 0;  i < vsrc.size();  ++i)
        {
            vsrc[i] = random_values[i];
        }
        median_rep_driver(vsrc, tmg_reps, "random", results);
    }

    printf("\nname, size, stl, avx, speedup, reps\n");
    for (auto const& e : results)
    {
        printf(e.c_str());
        printf("\n");
    }
}

/*
void
tf06()
{
    load_random_values();

    vector<float>  vsrc, vdst_avx, vdst_stl;
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
    fill(begin(vdst_stl), end(vdst_stl), 99.0f);
    fill(begin(vdst_avx), end(vdst_avx), 99.0f);

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
    fill(begin(vdst_stl), end(vdst_stl), 99.0f);
    fill(begin(vdst_avx), end(vdst_avx), 99.0f);

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

void tf07()
{
    mkl_domain_set_num_threads(1, MKL_DOMAIN_VML);

    int const       REPS = 100;
    size_t const    len  = 100'000;

    float           krnl[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    vector<float>   input(len), result1(len), result2(len);
    MKL_INT const   dsize = (MKL_INT) len;
    MKL_INT const   ksize = 3;
    MKL_INT const   start = ksize/2;
    int             status;

    VSLConvTaskPtr  ptask;
    stopwatch       sw;
    int64_t         mkl_time;
    int64_t         avx_time;

    for (size_t i = 0;  i < (len);  ++i)
    {
        input[i] = i + 1;
    }
    krnl[start] = 1.0f;

    status = vslsConvNewTask1D(&ptask, VSL_CONV_MODE_DIRECT, ksize, dsize, dsize);
    status = vslConvSetStart(ptask, &start);
    status = vslsConvExec1D(ptask, krnl, 1, input.data(), 1, result1.data(), 1);

    sw.start();
    for (int i = 0;  i < REPS;  ++i)
    {
        status = vslsConvExec1D(ptask, krnl, 1, input.data(), 1, result1.data(), 1);

    }
    sw.stop();
    mkl_time = sw.microseconds_elapsed()/REPS;

    sw.start();
    for (int i = 0;  i < REPS;  ++i)
    {
        avx_symm_convolve(result2.data(), krnl, ksize, input.data(), input.size());
    }
    sw.stop();
    avx_time = sw.microseconds_elapsed()/REPS;

    for (size_t i = 0;  i < (len - 16);  ++i)
    {
        if (result2[i] != result1[i])
        {
            printf("error at index %lu,  r1 = %.1f  r2 = %.1f\n", i, result1[i], result2[i]);
        }
    }

    for (size_t i = 0;  i < 32u;  ++i)
    {
        printf("[%02lu] in = %.1f  r1 = %.1f  r2 = %.1f\n", i, input[i], result1[i], result2[i]);
    }

    printf("\nfor convolution with:\n");
    printf("  n = %lu  k = %i   mkl = %ld  avx = %ld (usec)\n", len, ksize, mkl_time, avx_time);
    printf("\n");
    fflush(stdout);
}

void
conv_driver
(float const* pkrnl, size_t klen, float const* psrc, size_t len, size_t reps, char const* name, vector<string>& results)
{
    vector<float>   result1(round_up(len, 16u)), result2(round_up(len, 16u));
    MKL_INT const   dsize = (MKL_INT) len;
    MKL_INT const   ksize = klen;
    MKL_INT const   start = klen/2;
    int             status;

    VSLConvTaskPtr  ptask;
    stopwatch       sw;
    int64_t         mkl_time;
    int64_t         avx_time;
    double          speedup;
    char            resbuf[256];

    reps = 100'000'000/len;

    status = vslsConvNewTask1D(&ptask, VSL_CONV_MODE_DIRECT, ksize, dsize, dsize);
    status = vslConvSetStart(ptask, &start);
    status = vslsConvExec1D(ptask, pkrnl, 1, psrc, 1, result1.data(), 1);

    sw.start();
    for (size_t i = 0;  i < reps;  ++i)
    {
        status = vslsConvExec1D(ptask, pkrnl, 1, psrc, 1, result1.data(), 1);

    }
    sw.stop();
    mkl_time = sw.nanoseconds_elapsed()/reps;

    vslConvDeleteTask(&ptask);

    sw.start();
    for (size_t i = 0;  i < reps;  ++i)
    {
        avx_symm_convolve(result2.data(), pkrnl, ksize, psrc, len);
    }
    sw.stop();
    avx_time = sw.nanoseconds_elapsed()/reps;

    for (size_t i = 0, err = 0;  i < (len - 16);  ++i)
    {
        if (result2[i] != result1[i])
        {
            printf("error k = %lu  idx = %lu,  r1 = %5.1f  r2 = %5.1f\n", klen, i, result1[i], result2[i]);
            if (++err > 9) return;
        }
    }

    speedup = (double) mkl_time / (double) avx_time;
    printf("%s  n = %8lu  k = %2i  mkl = %9ld  avx = %8ld (ns)  s = %4.1f  r = %lu\n",
           name, len, ksize, mkl_time, avx_time, speedup, reps);
    fflush(stdout);

    sprintf(resbuf, "%s, %lu, %i, %ld, %ld, %.2f, %lu",
            name, len, ksize, mkl_time, avx_time, speedup, reps);
    results.push_back(string(resbuf));
}


void
tf08()
{
    load_random_values();
    mkl_domain_set_num_threads(1, MKL_DOMAIN_VML);

    vector<float>  vsrc;
    vector<float>  krnl;
    vector<string> results;
    size_t const        min_ncnt = 1000u;
    size_t const        max_ncnt = 10'000'000u;
    size_t const        tmg_reps = 1000;
    size_t const        ramp_mod = 1000;

    vsrc.reserve(max_ncnt);
    krnl.reserve(16);

    for (size_t klen = 3;  klen <= 15;  klen +=2)
    {
        krnl.resize(klen);

        for (size_t ncnt = min_ncnt;  ncnt <= max_ncnt;  ncnt *= 10)
        {
            vsrc.resize(ncnt);

            fill(krnl.begin(), krnl.end(), 0.0f);
            krnl[klen/2] = 1.0;

            for (size_t i = 0;  i < vsrc.size();  ++i)
            {
                vsrc[i] = i % ramp_mod;
            }
            conv_driver(krnl.data(), klen, vsrc.data(), ncnt, tmg_reps, "del-ramped", results);

            for (size_t i = 0;  i < vsrc.size();  ++i)
            {
                vsrc[i] = random_values[i];
            }
            conv_driver(krnl.data(), klen, vsrc.data(), ncnt, tmg_reps, "del-random", results);

            fill(krnl.begin(), krnl.end(), 1.0f);

            for (size_t i = 0;  i < vsrc.size();  ++i)
            {
                vsrc[i] = i % ramp_mod;
            }
            conv_driver(krnl.data(), klen, vsrc.data(), ncnt, tmg_reps, "sum-ramped", results);

            for (size_t i = 0;  i < vsrc.size();  ++i)
            {
                vsrc[i] = random_values[i];
            }
            conv_driver(krnl.data(), klen, vsrc.data(), ncnt, tmg_reps, "sum-random", results);
        }
        printf("\n");
        fflush(stdout);
    }

    printf("\nname, size, ksize, mkl, avx, speedup, reps\n");
    for (auto const& e : results)
    {
        printf(e.c_str());
        printf("\n");
    }
}
