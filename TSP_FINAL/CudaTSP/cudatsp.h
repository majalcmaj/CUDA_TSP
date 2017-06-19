#ifndef __CUDATSP_H
#define __CUDATSP_H

#include <stdlib.h>
#include "../common.h"

int run_cuda_tsp(location_t* locations, int locations_size, int on_device);

#endif
