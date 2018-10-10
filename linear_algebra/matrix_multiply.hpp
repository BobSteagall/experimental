#ifndef MATRIX_MULTIPLY_HPP_DEFINED
#define MATRIX_MULTIPLY_HPP_DEFINED

#include "matrix.hpp"

namespace std::experimental::la {

template<bool ISMUL, class E1, class E2>
struct engine_promotion;


//- engine * scalar cases.
//
template<bool ISMUL, class T1, class A1, class T2>
struct engine_promotion<ISMUL, dyn_matrix_engine<T1, A1>, T2>
{
    static_assert(is_matrix_element_v<T2>);

    using element_type = element_promotion_t<T1, T2>;
    using alloc_type   = typename std::allocator_traits<A1>::template rebind_alloc<element_type>;
    using engine_type  = dyn_matrix_engine<element_type, alloc_type>;
};

template<bool ISMUL, class T1, size_t R1, size_t C1, class T2>
struct engine_promotion<ISMUL, fs_matrix_engine<T1, R1, C1>, T2>
{
    static_assert(is_matrix_element_v<T2>);

    using element_type = element_promotion_t<T1, T2>;
    using engine_type  = fs_matrix_engine<element_type, R1, C1>;
};


//- engine * engine cases.  Note that in the cases where allocators are rebound, there is an
//  assumption that all allocators are standard-conformant.
//
template<bool ISMUL, class T1, class A1, class T2, class A2>
struct engine_promotion<ISMUL, dyn_matrix_engine<T1, A1>, dyn_matrix_engine<T2, A2>>
{
    using element_type = element_promotion_t<T1, T2>;
    using alloc_type   = typename std::allocator_traits<A1>::template rebind_alloc<element_type>;
    using engine_type  = dyn_matrix_engine<element_type, alloc_type>;
};

template<bool ISMUL, class T1, class A1, class T2, size_t R2, size_t C2>
struct engine_promotion<ISMUL, dyn_matrix_engine<T1, A1>, fs_matrix_engine<T2, R2, C2>>
{
    using element_type = element_promotion_t<T1, T2>;
    using alloc_type   = typename std::allocator_traits<A1>::template rebind_alloc<element_type>;
    using engine_type  = dyn_matrix_engine<element_type, alloc_type>;
};

template<bool ISMUL, class T1, size_t R1, size_t C1, class T2, class A2>
struct engine_promotion<ISMUL, fs_matrix_engine<T1, R1, C1>, dyn_matrix_engine<T2, A2>>
{
    using element_type = element_promotion_t<T1, T2>;
    using alloc_type   = typename std::allocator_traits<A2>::template rebind_alloc<element_type>;
    using engine_type  = dyn_matrix_engine<element_type, alloc_type>;
};

template<class T1, size_t R1, size_t C1, class T2, size_t R2, size_t C2>
struct engine_promotion<true, fs_matrix_engine<T1, R1, C1>, fs_matrix_engine<T2, R2, C2>>
{
    static_assert(C1 == R2);
    using element_type = element_promotion_t<T1, T2>;
    using engine_type  = fs_matrix_engine<element_type, R1, C2>;
};


template<class T1, size_t R1, size_t C1, class T2, size_t R2, size_t C2>
struct engine_promotion<false, fs_matrix_engine<T1, R1, C1>, fs_matrix_engine<T2, R2, C2>>
{
    static_assert(R1 == R2);
    static_assert(C1 == C2);
    using element_type = element_promotion_t<T1, T2>;
    using engine_type  = fs_matrix_engine<element_type, R1, C2>;
};


//- Alias interface to trait.
//
template<bool ISMUL, class E1, class E2>
using engine_promotion_t = typename engine_promotion<ISMUL, E1, E2>::engine_type;


//=================================================================================================
//  Traits type that performs engine promotion type computations for multiplication.
//=================================================================================================
//
template<class E1, class E2>
struct engine_mul_promotion;

//- engine * scalar cases.
//
template<class T1, class A1, class T2>
struct engine_mul_promotion<dyn_matrix_engine<T1, A1>, T2>
{
    static_assert(is_matrix_element_v<T2>);

    using element_type = element_promotion_t<T1, T2>;
    using alloc_type   = typename std::allocator_traits<A1>::template rebind_alloc<element_type>;
    using engine_type  = dyn_matrix_engine<element_type, alloc_type>;
};

template<class T1, size_t R1, size_t C1, class T2>
struct engine_mul_promotion<fs_matrix_engine<T1, R1, C1>, T2>
{
    static_assert(is_matrix_element_v<T2>);

    using element_type = element_promotion_t<T1, T2>;
    using engine_type  = fs_matrix_engine<element_type, R1, C1>;
};


//- engine * engine cases.  Note that in the cases where allocators are rebound, there is an
//  assumption that all allocators are standard-conformant.
//
template<class T1, class A1, class T2, class A2>
struct engine_mul_promotion<dyn_matrix_engine<T1, A1>, dyn_matrix_engine<T2, A2>>
{
    using element_type = element_promotion_t<T1, T2>;
    using alloc_type   = typename std::allocator_traits<A1>::template rebind_alloc<element_type>;
    using engine_type  = dyn_matrix_engine<element_type, alloc_type>;
};

template<class T1, class A1, class T2, size_t R2, size_t C2>
struct engine_mul_promotion<dyn_matrix_engine<T1, A1>, fs_matrix_engine<T2, R2, C2>>
{
    using element_type = element_promotion_t<T1, T2>;
    using alloc_type   = typename std::allocator_traits<A1>::template rebind_alloc<element_type>;
    using engine_type  = dyn_matrix_engine<element_type, alloc_type>;
};

template<class T1, size_t R1, size_t C1, class T2, class A2>
struct engine_mul_promotion<fs_matrix_engine<T1, R1, C1>, dyn_matrix_engine<T2, A2>>
{
    using element_type = element_promotion_t<T1, T2>;
    using alloc_type   = typename std::allocator_traits<A2>::template rebind_alloc<element_type>;
    using engine_type  = dyn_matrix_engine<element_type, alloc_type>;
};

template<class T1, size_t R1, size_t C1, class T2, size_t R2, size_t C2>
struct engine_mul_promotion<fs_matrix_engine<T1, R1, C1>, fs_matrix_engine<T2, R2, C2>>
{
    static_assert(C1 == R2);
    using element_type = element_promotion_t<T1, T2>;
    using engine_type  = fs_matrix_engine<element_type, R1, C2>;
};


//- Alias interface to trait.
//
template<class E1, class E2>
using engine_multiply_t = typename engine_mul_promotion<E1, E2>::engine_type;


//=================================================================================================
//  Traits type that performs the multiplications.  Optimized multiplication can be implemented
//  by providing partial/full specializations.
//=================================================================================================
//
template<class OP1, class OP2>
struct matrix_multiply_traits;

//- vector/scalar
//
template<class E1, class T2>
struct matrix_multiply_traits<col_vector<E1>, T2>
{
    using engine_type = engine_multiply_t<E1, T2>;
    using result_type = col_vector<engine_type>;

    static result_type  multiply(col_vector<E1> const& m, T2 s);
};

template<class E1, class T2>
struct matrix_multiply_traits<row_vector<E1>, T2>
{
    using engine_type = engine_multiply_t<E1, T2>;
    using result_type = col_vector<engine_type>;

    static result_type  multiply(row_vector<E1> const& m, T2 s);
};


//- matrix/scalar
//
template<class E1, class T2>
struct matrix_multiply_traits<matrix<E1>, T2>
{
    using engine_type = engine_multiply_t<E1, T2>;
    using result_type = col_vector<engine_type>;

    static result_type  multiply(matrix<E1> const& m, T2 s);
};


//- vector/vector
//
template<class E1, class E2>
struct matrix_multiply_traits<row_vector<E1>, col_vector<E2>>
{
    using elem_type_1 = typename row_vector<E1>::element_type;
    using elem_type_2 = typename col_vector<E2>::element_type;
    using result_type = element_promotion_t<elem_type_1, elem_type_2>;

    static result_type  multiply(row_vector<E1> const& rv, col_vector<E1> const& cv);
};

template<class E1, class E2>
struct matrix_multiply_traits<col_vector<E1>, row_vector<E2>>
{
    using engine_type = engine_multiply_t<E1, E2>;
    using result_type = matrix<engine_type>;

    static result_type  multiply(col_vector<E1> const& cv, row_vector<E1> const& rv);
};


//- matrix/vector
//
template<class E1, class E2>
struct matrix_multiply_traits<matrix<E1>, col_vector<E2>>
{
    using engine_type = engine_multiply_t<E1, E2>;
    using result_type = col_vector<engine_type>;

    static result_type  multiply(matrix<E1> const& m, col_vector<E2> const& cv);
};

template<class E1, class E2>
struct matrix_multiply_traits<matrix<E1>, row_vector<E2>>
{
    using engine_type = engine_multiply_t<E1, E2>;
    using result_type = matrix<engine_type>;

    static result_type  multiply(matrix<E1> const& m, row_vector<E2> const& cv);
};


//- vector/matrix
//
template<class E1, class E2>
struct matrix_multiply_traits<col_vector<E1>, matrix<E2>>
{
    using engine_type = engine_multiply_t<E1, E2>;
    using result_type = matrix<engine_type>;

    static result_type  multiply(col_vector<E1> const& rv, matrix<E2> const& m);
};

template<class E1, class E2>
struct matrix_multiply_traits<row_vector<E1>, matrix<E2>>
{
    using engine_type = engine_multiply_t<E1, E2>;
    using result_type = row_vector<engine_type>;

    static result_type  multiply(row_vector<E1> const& rv, matrix<E2> const& m);
};


//- matrix/matrix
//
template<class E1, class E2>
struct matrix_multiply_traits<matrix<E1>, matrix<E2>>
{
    using engine_type = engine_multiply_t<E1, E2>;
    using result_type = matrix<engine_type>;

    static result_type  multiply(matrix<E1> const& m1, matrix<E2> const& m2);
};


//=================================================================================================
//  Multiplication operators, which forward to the traits types that perform the multiplications.
//=================================================================================================
//
//- col_vector/scalar
//
template<class E1, class E2>
inline auto
operator *(col_vector<E1> const& cv, E2 s)
{
    return matrix_multiply_traits<col_vector<E1>, E2>::multiply(cv, s);
}

template<class E1, class E2>
inline auto
operator *(E1 s, col_vector<E2> const& cv)
{
    return matrix_multiply_traits<col_vector<E2>, E1>::multiply(cv, s);
}

//- row_vector/scalar
//
template<class E1, class E2>
inline auto
operator *(row_vector<E1> const& rv, E2 s)
{
    return matrix_multiply_traits<row_vector<E1>, E2>::multiply(rv, s);
}

template<class E1, class E2>
inline auto
operator *(E1 s, row_vector<E2> const& rv)
{
    return matrix_multiply_traits<row_vector<E2>, E1>::multiply(rv, s);
}

//- matrix/scalar
//
template<class E1, class E2>
inline auto
operator *(matrix<E1> const& m, E2 s)
{
    return matrix_multiply_traits<matrix<E1>, E2>::multiply(m, s);
}

template<class E1, class E2>
inline auto
operator *(E1 s, matrix<E2> const& m)
{
    return matrix_multiply_traits<matrix<E2>, E1>::multiply(m, s);
}

//- vector/vector
//
template<class E1, class E2>
inline auto
operator *(row_vector<E1> const& rv, col_vector<E2> const& cv)
{
    return matrix_multiply_traits<row_vector<E1>, col_vector<E2>>::multiply(rv, cv);
}

template<class E1, class E2>
inline auto
operator *(col_vector<E1> const& cv, row_vector<E2> const& rv)
{
    return matrix_multiply_traits<col_vector<E1>, row_vector<E2>>::multiply(cv, rv);
}

//- matrix/vector
//
template<class E1, class E2>
inline auto
operator *(matrix<E1> const& m, col_vector<E2> const& cv)
{
    return matrix_multiply_traits<matrix<E1>, col_vector<E2>>::multiply(m, cv);
}

template<class E1, class E2>
inline auto
operator *(matrix<E1> const& m, row_vector<E2> const& rv)
{
    return matrix_multiply_traits<matrix<E1>, row_vector<E2>>::multiply(m, rv);
}

//- vector/matrix
//
template<class E1, class E2>
inline auto
operator *(col_vector<E1> const& cv, matrix<E2> const& m)
{
    return matrix_multiply_traits<col_vector<E1>, matrix<E2>>::multiply(cv, m);
}

template<class E1, class E2>
inline auto
operator *(row_vector<E1> const& rv, matrix<E2> const& m)
{
    return matrix_multiply_traits<row_vector<E1>, matrix<E2>>::multiply(rv, m);
}

//- matrix/matrix
//
template<class E1, class E2>
inline auto
operator *(matrix<E1> const& m1, matrix<E2> const& m2)
{
    return matrix_multiply_traits<matrix<E1>, matrix<E2>>::multiply(m1, m2);
}

}       //- std::experimental::la namespace
#endif  //- MATRIX_MULTIPLY_HPP_DEFINED
