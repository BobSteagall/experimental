#include <complex>
#include "linear_algebra.hpp"

using cx_float  = std::complex<float>;
using cx_double = std::complex<double>;
using namespace std::experimental::la;

//- Some detection idiom stuff to make sure SFINAE is working for fixed-size -versus-
//  dynamic interfaces.
//
template<typename T, typename = void>
struct has_resize : std::false_type {};

template<typename T>
struct has_resize<T, std::void_t<decltype(std::declval<T>().resize(0,0))>> 
:   std::true_type {};

template<typename T>
inline constexpr bool   has_resize_v = has_resize<T>::value;

void    t01()
{
    constexpr bool  b0 = is_complex_v<std::string>;
    constexpr bool  b1 = is_complex_v<double>;
    constexpr bool  b2 = is_complex_v<std::complex<int>>;

    constexpr bool  b4 = is_matrix_element_v<double>;
    constexpr bool  b5 = is_matrix_element_v<std::complex<double>>;
    constexpr bool  b6 = is_matrix_element_v<std::complex<int32_t>>;

    constexpr bool  b001 = is_matrix_element_v<std::string>;
    constexpr bool  b002 = is_matrix_element_v<std::complex<std::string>>;

    //- use detection idiom stuff from above.
    //
    constexpr bool  b003 = has_resize_v<fs_matrix<double, 3, 3>>;
    constexpr bool  b004 = has_resize_v<dyn_matrix<double>>;
}

void t02()
{
    fs_matrix_engine<double, 2, 2>      e22;
    fs_matrix_engine<cx_double, 3, 3>   e33;

    dyn_matrix_engine<double>       de2;
    dyn_matrix_engine<cx_double>    de3;

    matrix_transpose_engine<fs_matrix_engine<cx_double, 3, 3>>  te2(e33);
    matrix_transpose_engine<dyn_matrix_engine<cx_double>>       te3(de3);

#ifndef ENFORCE_COMPLEX_OPERAND_HOMOGENEITY
    element_promotion_t<int32_t, cx_double>     v1 = 0;
    element_promotion_t<cx_float, double>       v2 = 0;
    element_promotion_t<double, cx_float>       v3 = 0;
    element_promotion_t<cx_float, cx_double>    v4 = 0;
#endif
}

void t03()
{
    fs_col_vector<double, 3>    fcv1;
    fs_row_vector<double, 3>    frv1;
    fs_matrix<double, 3, 3>     fm1;

    dyn_col_vector<double>      dcv1(16);
    dyn_row_vector<double>      drv1(16);
    dyn_matrix<double>          dmd1(16, 16);
}

void t04()
{
    float       f = 1.0f;
    double      d = 1.0;
    cx_double   c = {1.0, 0.0};

    dyn_matrix<float>       mf(3, 3);
    dyn_matrix<double>      md(3, 3);
    dyn_matrix<cx_double>   mc(3, 3);

    auto    m01 = mf * f;
    auto    m02 = md * d;
    auto    m03 = mc * c;
    auto    m04 = mf * d;
    auto    m05 = md * f;

    auto    m11 = f * mf;
    auto    m12 = d * md;
    auto    m13 = c * mc;
    auto    m14 = d * mf;
    auto    m15 = f * md;

    auto    m21 = mf * mf;
    auto    m22 = md * md;
    auto    m23 = mc * mc;
    auto    m24 = md * mf;
    auto    m25 = mf * md;
}

void t05()
{
    float       f = 1.0f;
    double      d = 1.0;
    cx_double   c = {1.0, 0.0};

    fs_matrix<float, 3, 3>      mf;
    fs_matrix<double, 3, 3>     md;
    fs_matrix<cx_double, 3, 3>  mc;

    auto    m01 = mf * f;
    auto    m02 = md * d;
    auto    m03 = mc * c;
    auto    m04 = mf * d;
    auto    m05 = md * f;

    auto    m11 = f * mf;
    auto    m12 = d * md;
    auto    m13 = c * mc;
    auto    m14 = d * mf;
    auto    m15 = f * md;

    auto    m21 = mf * mf;
    auto    m22 = md * md;
    auto    m23 = mc * mc;
    auto    m24 = md * mf;
    auto    m25 = mf * md;

    fs_matrix<double, 3, 7>     md2;
    fs_matrix<float, 7, 5>      md3;

    auto    m31 = md2 * md3;
}

void t06()
{
    float       f = 1.0f;
    double      d = 1.0;
    cx_double   c = {1.0, 0.0};

    dyn_matrix<float>       dmf(3, 3);
    dyn_matrix<double>      dmd(3, 3);
    dyn_matrix<cx_double>   dmc(3, 3);

    fs_matrix<float, 3, 3>      fmf;
    fs_matrix<double, 3, 3>     fmd;
    fs_matrix<cx_double, 3, 3>  fmc;

    auto    m01 = dmf*fmf;
    auto    m02 = dmd*fmd;
    auto    m03 = dmc*fmc;
    auto    m04 = fmf*dmf;
    auto    m05 = fmd*dmd;
    auto    m06 = fmc*dmc;
}

void t07()
{
    float       f = 1.0f;
    double      d = 1.0;
    cx_double   c = {1.0, 0.0};

    dyn_col_vector<float>       dcvf(3);
    dyn_col_vector<double>      dcvd(3);
    dyn_col_vector<cx_double>   dcvc(3);

    fs_col_vector<float, 3>     fcvf;
    fs_col_vector<double, 3>    fcvd;
    fs_col_vector<cx_double, 3> fcvc;

    auto    r01 = dcvf * f;
    auto    r02 = dcvd * d;
    auto    r03 = dcvc * c;
    auto    r04 = dcvf * d;
    auto    r05 = dcvd * f;

    auto    r11 = f * dcvf;
    auto    r12 = d * dcvd;
    auto    r13 = c * dcvc;
    auto    r14 = d * dcvf;
    auto    r15 = f * dcvd;

    auto    r21 = fcvf * f;
    auto    r22 = fcvd * d;
    auto    r23 = fcvc * c;
    auto    r24 = fcvf * d;
    auto    r25 = fcvd * f;

    auto    r31 = f * fcvf;
    auto    r32 = d * fcvd;
    auto    r33 = c * fcvc;
    auto    r34 = d * fcvf;
    auto    r35 = f * fcvd;
}

void t08()
{
    float       f = 1.0f;
    double      d = 1.0;
    cx_double   c = {1.0, 0.0};

    dyn_row_vector<float>       drvf(3);
    dyn_row_vector<double>      drvd(3);
    dyn_row_vector<cx_double>   drvc(3);

    fs_row_vector<float, 3>     frvf;
    fs_row_vector<double, 3>    frvd;
    fs_row_vector<cx_double, 3> frvc;

    auto    r01 = drvf * f;
    auto    r02 = drvd * d;
    auto    r03 = drvc * c;
    auto    r04 = drvf * d;
    auto    r05 = drvd * f;

    auto    r11 = f * drvf;
    auto    r12 = d * drvd;
    auto    r13 = c * drvc;
    auto    r14 = d * drvf;
    auto    r15 = f * drvd;

    auto    r21 = frvf * f;
    auto    r22 = frvd * d;
    auto    r23 = frvc * c;
    auto    r24 = frvf * d;
    auto    r25 = frvd * f;

    auto    r31 = f * frvf;
    auto    r32 = d * frvd;
    auto    r33 = c * frvc;
    auto    r34 = d * frvf;
    auto    r35 = f * frvd;
}

void t09()
{
    dyn_col_vector<float>       dcvf(3);
    dyn_col_vector<double>      dcvd(3);

    fs_col_vector<float, 3>     fcvf;
    fs_col_vector<double, 3>    fcvd;

    dyn_row_vector<float>       drvf(3);
    dyn_row_vector<double>      drvd(3);

    fs_row_vector<float, 3>     frvf;
    fs_row_vector<double, 3>    frvd;

    auto    r01 = drvf * dcvf;
    auto    r02 = frvf * dcvf;
    auto    r03 = drvf * fcvf;
    auto    r04 = frvf * fcvf;

    auto    r11 = dcvf * drvf;
    auto    r12 = fcvf * drvf;
    auto    r13 = dcvf * frvf;
    auto    r14 = fcvf * frvf;

    auto    r21 = drvf * dcvd;
    auto    r22 = frvf * dcvd;
    auto    r23 = drvf * fcvd;
    auto    r24 = frvf * fcvd;

    auto    r31 = dcvf * drvd;
    auto    r32 = fcvf * drvd;
    auto    r33 = dcvf * frvd;
    auto    r34 = fcvf * frvd;
}

void t10()
{
    dyn_col_vector<float>       dcvf(3, 3);
    dyn_col_vector<double>      dcvd(3, 3);

    fs_col_vector<float, 3>     fcvf;
    fs_col_vector<double, 3>    fcvd;

    dyn_row_vector<float>       drvf(3, 3);
    dyn_row_vector<double>      drvd(3, 3);

    fs_row_vector<float, 3>     frvf;
    fs_row_vector<double, 3>    frvd;

    dyn_matrix<float>           dmf(3, 3);
    dyn_matrix<double>          dmd(3, 3);

    fs_matrix<float, 3, 3>      fmf;
    fs_matrix<float, 1, 3>      fmf_rv;
    fs_matrix<float, 3, 1>      fmf_cv;
    fs_matrix<double, 3, 3>     fmd;

    auto    r01 = dmf * dcvf;
    auto    r02 = dmf * drvf;
    auto    r03 = drvf * dmf;
    auto    r04 = dcvf * dmf;

    auto    r11 = dmf * dcvd;
    auto    r12 = dmf * drvd;
    auto    r13 = drvf * dmd;
    auto    r14 = dcvf * dmd;

    auto    r21 = fmf * fcvf;
    auto    r22 = fmf_cv * frvf;
    auto    r23 = frvf * fmf;
    auto    r24 = fcvf * fmf_rv;

    auto    r31 = fmf * fcvd;
    auto    r32 = fmf_cv * frvd;
    auto    r33 = frvf * fmd;
    auto    r34 = fcvd * fmf_rv;
}

int main()
{
    return 0;
}
