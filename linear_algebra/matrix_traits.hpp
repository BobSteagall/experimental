#ifndef MATRIX_TRAITS_HPP_DEFINED
#define MATRIX_TRAITS_HPP_DEFINED

#include "matrix_engines.hpp"

namespace std::experimental::la {

template<class E1, class E2>
struct element_promotion_helper
{
    static_assert(std::is_arithmetic_v<E1> && std::is_arithmetic_v<E2>);
    using type = decltype(E1() * E2());
};

template<class E1, class E2>
using element_promotion_helper_t = typename element_promotion_helper<E1, E2>::type;


template<class E1, class E2>
struct element_promotion
{
    using type = element_promotion_helper_t<E1, E2>;
};

template<class E1, class E2>
struct element_promotion<E1, std::complex<E2>>
{
    using type = std::complex<element_promotion_helper_t<E1, E2>>;
};

template<class E1, class E2>
struct element_promotion<std::complex<E1>, E2>
{
    using type = std::complex<element_promotion_helper_t<E1, E2>>;
};

template<class E1, class E2>
struct element_promotion<std::complex<E1>, std::complex<E2>>
{
    using type = std::complex<element_promotion_helper_t<E1, E2>>;
};

template<class E1, class E2>
using element_promotion_t = typename element_promotion<E1, E2>::type;


}       //- std::experimental::la namespace
#endif  //- MATRIX_TRAITS_HPP_DEFINED
