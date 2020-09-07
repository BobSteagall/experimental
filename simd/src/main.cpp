#include "test.hpp"
//#include <mkl.h>
#include <pthread.h>

int main()
{
    pin_thread();
    load_random_values();

    tf01();
    tf02();
    tf03();
    tf04();
    tf05();

    return 0;
}
