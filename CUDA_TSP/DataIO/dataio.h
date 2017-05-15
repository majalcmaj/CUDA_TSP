#ifndef __DATAIO_H
#define __DATAIO_H

#include "../common.h"
#include<string.h>

int read_csv(const char* path, location_t** locations_out, int* length_out);
int write_csv(const char* path, location_t* locations, int locations_count);

#endif //!__DATAIO_H
