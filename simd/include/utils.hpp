#ifndef KEWB_UTILS_HPP_DEFINED
#define KEWB_UTILS_HPP_DEFINED

#include <cstdio>
#include <cstdint>

#include <chrono>
#include <string>
#include <type_traits>

template<class T, class U>
inline T
round_up(T val, U radix)
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>);

    T   rem = val % static_cast<T>(radix);
    return (rem == 0) ? val : val - rem + static_cast<T>(radix);
}


template<class T, class U>
inline T
round_down(T val, U radix)
{
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>);

    T   rem = val % static_cast<T>(radix);
    return (rem == 0) ? val : val - rem;
}


class stopwatch
{
  public:
    ~stopwatch() = default;

    stopwatch()
    :   m_start(clock_type::now())
    ,   m_stop(m_start)
    {}

    int64_t
    microseconds_elapsed() const
    {
        return static_cast<int64_t>(std::chrono::duration_cast<usecs>(m_stop - m_start).count());
    }

    int64_t
    milliseconds_elapsed() const
    {
        return static_cast<int64_t>(std::chrono::duration_cast<msecs>(m_stop - m_start).count());
    }

    int64_t
    seconds_elapsed() const
    {
        return static_cast<int64_t>(std::chrono::duration_cast<secs>(m_stop - m_start).count());
    }

    void
    start()
    {
        m_stop = m_start = clock_type::now();
    }

    void
    stop()
    {
        m_stop = clock_type::now();
    }

  private:
    using clock_type = std::chrono::system_clock;
    using time_point = std::chrono::time_point<clock_type>;
    using usecs      = std::chrono::microseconds;
    using msecs      = std::chrono::milliseconds;
    using secs       = std::chrono::seconds;

    time_point  m_start;
    time_point  m_stop;
};

#endif  //- KEWB_UTILS_HPP_DEFINED
