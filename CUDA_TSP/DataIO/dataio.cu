#include "dataio.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define INITIAL_LOCATIONS_ARRAY_SIZE 1024
#define MAX_LINE_LENGTH 256
#define GROW_BUFFER_TIMES 2
#define CSV_DELIMITERS ",\n"
#define CSV_HEADER "locId,latitude,longitude\n"

int deserialize_location(location_t* location, char* str);
int grow_buffer(location_t** locations_buff, int* size);
int serialize_location(FILE* f, location_t* location);

int read_csv(const char* path, location_t** locations_out, int* length_out)
{
	int retcode;
	FILE* ds;
	int max_buff_length = INITIAL_LOCATIONS_ARRAY_SIZE;
	int currindex = 0;
	location_t* location_buff = (location_t*)malloc(sizeof(location_t) * max_buff_length);
	location_t location;
	char linebuff[MAX_LINE_LENGTH];

	ds = fopen(path, "r");
	if(ds == NULL) {
		retcode = 1;
		perror("Err while opening the file: ");
		goto failure;
	}
	if(fgets(linebuff, MAX_LINE_LENGTH, ds) == NULL) {
		retcode = 6;
		goto failure;
	}

	if(strcmp(linebuff, CSV_HEADER) != 0) {
		retcode = 8;
		goto failure;
	}

	while(fgets(linebuff, MAX_LINE_LENGTH, ds) != NULL) {
		retcode = deserialize_location(&location, linebuff);
		if(retcode != 0)
			goto failure;
		if(currindex >= max_buff_length) {
			retcode = grow_buffer(&location_buff, &max_buff_length);
			if(retcode != 0)
				goto failure;
		}
		location_buff[currindex] = location;
		currindex += 1;
	}
	if(!feof(ds))
	{
		perror("Not whole file has been parsed");
		goto failure;
	}
	fclose(ds);
	*locations_out = location_buff;
	*length_out = currindex;
	return 0;

failure:
	if(ds != NULL)
		fclose(ds);
	if(location_buff != NULL)
		free(location_buff);
	return retcode;
}

int write_csv(const char* path, location_t* locations, int locations_count)
{
	FILE* f = fopen(path, "w");
	if(f == NULL)
	{
		return 1;
	}
	fprintf(f, CSV_HEADER);
	for(int i = 0 ; i < locations_count ; i ++)
		serialize_location(f, &locations[i]);
	fclose(f);
	return 0;
}

int deserialize_location(location_t* location, char* str)
{
	char* token;
	char* end;
	token = (char*)strtok(str, CSV_DELIMITERS);
	location->location_id = strtol(token, &end, 0);
	if(*end != '\0')
		return 2;
	token = (char*)strtok(NULL, CSV_DELIMITERS);
	location->latitude = strtof(token, &end);
	if(*end != '\0')
			return 3;
	token = (char*)strtok(NULL, CSV_DELIMITERS);
	location->longitude = strtof(token, &end);
	if(*end != '\0')
		return 4;
	return 0;
}

int grow_buffer(location_t** locations_buff, int* size)
{
	*size *= GROW_BUFFER_TIMES;
	*locations_buff = (location_t*)realloc(*locations_buff, *size * sizeof(location_t));
	if(locations_buff == NULL) {
		return 5;
	}
	return 0;
}

int serialize_location(FILE* f, location_t* location)
{
	fprintf(f, "%d,%lf%,lf\n", location->location_id, location->latitude, location->longitude);
	return 0;
}


