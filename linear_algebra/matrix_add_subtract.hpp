#ifndef MATRIX_ADD_SUBTRACT_HPP_DEFINED
#define MATRIX_ADD_SUBTRACT_HPP_DEFINED

#include "matrix.hpp"

namespace std::experimental::la {

//=================================================================================================
//
template<class E1, class E2>
struct engine_addsub_promotion;

template<class E1, class A1, class E2, class A2>
struct engine_addsub_promotion<dyn_matrix_engine<E1, A1>, dyn_matrix_engine<E2, A2>>
{
    using element_type = element_promotion_t<E1, E2>;
    using alloc_type   = typename std::allocator_traits<A1>::template rebind_alloc<element_type>;
    using engine_type  = dyn_matrix_engine<element_type, alloc_type>;
};

template<class E1, class A1, class E2, size_t R2, size_t C2>
struct engine_addsub_promotion<dyn_matrix_engine<E1, A1>, fs_matrix_engine<E2, R2, C2>>
{
    using element_type = element_promotion_t<E1, E2>;
    using alloc_type   = typename std::allocator_traits<A1>::template rebind_alloc<element_type>;
    using engine_type  = dyn_matrix_engine<element_type, alloc_type>;
};

template<class E1, size_t R1, size_t C1, class E2, class A2>
struct engine_addsub_promotion<fs_matrix_engine<E1, R1, C1>, dyn_matrix_engine<E2, A2>>
{
    using element_type = element_promotion_t<E1, E2>;
    using alloc_type   = typename std::allocator_traits<A2>::template rebind_alloc<element_type>;
    using engine_type  = dyn_matrix_engine<element_type, alloc_type>;
};

template<class E1, size_t R1, size_t C1, class E2, size_t R2, size_t C2>
struct engine_addsub_promotion<fs_matrix_engine<E1, R1, C1>, fs_matrix_engine<E2, R2, C2>>
{
    static_assert(R1 == R2);
    static_assert(C1 == C2);
    using element_type = element_promotion_t<E1, E2>;
    using engine_type  = fs_matrix_engine<element_type, R1, C1>;
};


template<class E1, class E2>
using engine_addsub_t = typename engine_addsub_promotion<E1, E2>::engine_type;


}       //- std::experimental::la namespace
#endif  //- MATRIX_ADD_SUBTRACT_HPP_DEFINED
