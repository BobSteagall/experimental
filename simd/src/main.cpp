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
//#if 0
    tf05();
//#else
    tf07();
    tf08();
//#endif
    return 0;
}
