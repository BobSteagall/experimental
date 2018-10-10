#ifndef LINEAR_ALGEBRA_FWD_HPP_DEFINED
#define LINEAR_ALGEBRA_FWD_HPP_DEFINED

#include <cstdint>
#include <complex>
#include <memory>
#include <tuple>
#include <type_traits>

//- New experimental namespace for test implementation
//
namespace std::experimental::la {

template<class ENG> class col_vector;
template<class ENG> class row_vector;
template<class ENG> class matrix;

}       //- std::experimental::la namespace
#endif  //- LINEAR_ALGEBRA_FWD_HPP_DEFINED
