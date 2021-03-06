#ifndef MATRIX_TRAITS_HPP_DEFINED
#define MATRIX_TRAITS_HPP_DEFINED

#include "linear_algebra_fwd.hpp"

namespace std::experimental::la {

//- New trait "is_complex" to detect std::complex.
//
template<class T>
struct is_complex : public false_type
{};

template<class T>
struct is_complex<std::complex<T>> 
:   public std::bool_constant<std::is_arithmetic_v<T>>
{};

template<class T>
inline constexpr bool   is_complex_v = is_complex<T>::value;


//- New trait "is_matrix_element" to detect valid matrix/vector element types.
//
template<class T>
struct is_matrix_element 
:   public std::bool_constant<std::is_arithmetic_v<T> || is_complex_v<T>>
{};

template<class T>
inline constexpr bool   is_matrix_element_v = is_matrix_element<T>::value;


//=================================================================================================
//- Helper traits for performing element type promotion.
//
template<class T1, class T2>
struct element_promotion_helper
{
    static_assert(std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>);
    using type = decltype(T1() * T2());
};

template<class T1, class T2>
using element_promotion_helper_t = typename element_promotion_helper<T1, T2>::type;


//- Unfortunately, in C++17, std::complex only permits arithmetical expressions between
//  homogeneous element types: e.g., complex<float>*complex<float> and float*complex<float>
//  are permitted but double*complex<float> is not.
//
#define ENFORCE_COMPLEX_OPERAND_HOMOGENEITY

#ifdef ENFORCE_COMPLEX_OPERAND_HOMOGENEITY
    #define ASSERT_COMPLEX_OPERAND_HOMOGENEITY(T1,T2)   static_assert(std::is_same_v<T1, T2>)
#else
    #define ASSERT_COMPLEX_OPERAND_HOMOGENEITY(T1,T2)
#endif

//- Traits to perform type promotion for arithmetical expressions, using the helper above.
//
template<class T1, class T2>
struct element_promotion
{
    using type = element_promotion_helper_t<T1, T2>;
};

template<class T1, class T2>
struct element_promotion<T1, std::complex<T2>>
{
    ASSERT_COMPLEX_OPERAND_HOMOGENEITY(T1,T2);
    using type = std::complex<element_promotion_helper_t<T1, T2>>;
};

template<class T1, class T2>
struct element_promotion<std::complex<T1>, T2>
{
    ASSERT_COMPLEX_OPERAND_HOMOGENEITY(T1,T2);
    using type = std::complex<element_promotion_helper_t<T1, T2>>;
};

template<class T1, class T2>
struct element_promotion<std::complex<T1>, std::complex<T2>>
{
    ASSERT_COMPLEX_OPERAND_HOMOGENEITY(T1,T2);
    using type = std::complex<element_promotion_helper_t<T1, T2>>;
};

template<class T1, class T2>
using element_promotion_t = typename element_promotion<T1, T2>::type;

}       //- std::experimental::la namespace
#endif  //- MATRIX_TRAITS_HPP_DEFINED
