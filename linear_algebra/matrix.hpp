#ifndef MATRIX_HPP_DEFINED
#define MATRIX_HPP_DEFINED

#include "matrix_engines.hpp"

namespace std::experimental::la {

//- New trait to use with SFINAE to determine if an engine is resizable; if it is, then
//  something can be enabled.
//
template<class ET1, class ET2>
using enable_if_resizable_t = 
    typename std::enable_if_t<is_same_v<ET1, ET2> && ET1::is_resizable_type::value, bool>;

//=================================================================================================
//  A column vector parametrized by an engine type.
//=================================================================================================
//
template<class ENG>
class col_vector
{
  public:
    using engine_type       = ENG;
    using element_type      = typename ENG::element_type;
    using is_resizable_type = typename ENG::is_resizable_type;
    using size_tuple        = typename ENG::size_tuple;
    using transpose_type    = row_vector<matrix_transpose_engine<ENG>>;

  public:
    ~col_vector() = default;
    col_vector();
    col_vector(col_vector&&) = default;
    col_vector(col_vector const&) = default;

    template<class ET2>
    col_vector(col_vector<ET2> const& src);

    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    col_vector(size_t rows);
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    col_vector(size_t rows, size_t rowcap);

    col_vector&     operator =(col_vector&&) = default;
    col_vector&     operator =(col_vector const&) = default;

    template<class ET2>
    col_vector&     operator =(col_vector<ET2> const& rhs);

    //- Const element access.
    //
    element_type        operator ()(size_t i) const;
    element_type const* data() const;

    //- Accessors.
    //
    size_t  columns() const noexcept;
    size_t  rows() const noexcept;
    size_t  size() const noexcept;

    size_t  column_capacity() const noexcept;
    size_t  row_capacity() const noexcept;
    size_t  capacity() const noexcept;

    //- Common functions.
    //
    transpose_type  tr() const;

    //- Mutable element access.
    //
    element_type    operator ()(size_t i);
    element_type*   data();

    //- Change capacity.
    //
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    void    reserve(size_t rowcap);

    //- Change size.
    //
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    void    resize(size_t rows);

    //- Change size and capacity in one shot.
    //
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    void    resize(size_t rows, size_t rowcap);

  private:
    engine_type     m_engine;
};


//=================================================================================================
//  A row vector parametrized by an engine type.
//=================================================================================================
//
template<class ENG>
class row_vector
{
  public:
    using engine_type       = ENG;
    using element_type      = typename ENG::element_type;
    using is_resizable_type = typename ENG::is_resizable_type;
    using size_tuple        = typename ENG::size_tuple;
    using transpose_type    = col_vector<matrix_transpose_engine<ENG>>;

  public:
    ~row_vector() = default;
    row_vector();
    row_vector(row_vector&&) = default;
    row_vector(row_vector const&) = default;

    template<class ET2>
    row_vector(row_vector<ET2> const& src);

    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    row_vector(size_t cols);
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    row_vector(size_t cols, size_t colcap);

    row_vector&     operator =(row_vector&&) = default;
    row_vector&     operator =(row_vector const&) = default;

    template<class ET2>
    row_vector&     operator =(row_vector<ET2> const& rhs);

    //- Const element access.
    //
    element_type        operator ()(size_t i) const;
    element_type const* data() const;

    //- Accessors.
    //
    size_t  columns() const noexcept;
    size_t  rows() const noexcept;
    size_t  size() const noexcept;

    size_t  column_capacity() const noexcept;
    size_t  row_capacity() const noexcept;
    size_t  capacity() const noexcept;

    //- Common functions.
    //
    transpose_type  tr() const;

    //- Mutable element access.
    //
    element_type    operator ()(size_t i);
    element_type*   data();

    //- Change capacity.
    //
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    void    reserve(size_t colcap);

    //- Change size.
    //
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    void    resize(size_t cols);

    //- Change size and capacity in one shot.
    //
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    void    resize(size_t cols, size_t colcap);

  private:
    engine_type     m_engine;
};


//=================================================================================================
//  A matrix parametrized by an engine type.
//=================================================================================================
//
template<class ENG>
class matrix
{
  public:
    using engine_type       = ENG;
    using element_type      = typename ENG::element_type;
    using is_resizable_type = typename ENG::is_resizable_type;
    using size_tuple        = typename ENG::size_tuple;
    using transpose_type    = matrix<matrix_transpose_engine<ENG>>;

  public:
    ~matrix() = default;
    matrix();
    matrix(matrix&&) = default;
    matrix(matrix const&) = default;

    template<class ET2>
    matrix(matrix<ET2> const& src);

    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    matrix(size_tuple size);
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    matrix(size_t rows, size_t cols);
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    matrix(size_t rows, size_t cols, size_t rowcap, size_t colcap);

    matrix&     operator =(matrix&&) = default;
    matrix&     operator =(matrix const&) = default;

    template<class ET2>
    matrix&     operator =(matrix<ET2> const& rhs);

    //- Const element access.
    //
    element_type        operator ()(size_t i, size_t j) const;
    element_type const* data() const;

    //- Accessors.
    //
    size_t      columns() const noexcept;
    size_t      rows() const noexcept;
    size_tuple  size() const noexcept;

    size_t      column_capacity() const noexcept;
    size_t      row_capacity() const noexcept;
    size_tuple  capacity() const noexcept;

    //- Common functions.
    //
    transpose_type  tr() const;

    //- Mutable element access.
    //
    element_type    operator ()(size_t i, size_t j);
    element_type*   data();

    //- Change capacity.
    //
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    void    reserve(size_tuple cap);
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    void    reserve(size_t rowcap, size_t colcap);

    //- Change size.
    //
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    void    resize(size_tuple size);
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    void    resize(size_t rows, size_t cols);

    //- Change size and capacity in one shot.
    //
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    void    resize(size_tuple size, size_tuple cap);
    template<class ET2 = ENG, enable_if_resizable_t<ENG, ET2> = true>
    void    resize(size_t rows, size_t cols, size_t rowcap, size_t colcap);

  private:
    engine_type     m_engine;
};

}       //- std::experimental::la namespace
#endif  //- MATRIX_HPP_DEFINED
