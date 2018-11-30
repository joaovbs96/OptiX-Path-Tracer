// optix code:
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "prd.h"
#include "sampling.h"

RT_PROGRAM void exception_program( void ) {
    const unsigned int code = rtGetExceptionCode();

    // if you want to treat each exception differently
    //if( code == RT_EXCEPTION_STACK_OVERFLOW )
    rtPrintExceptionDetails();
}