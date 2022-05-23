#ifndef SATCHECK_CUH
#define SATCHECK_CUH
#include <inttypes.h>
#include <stdio.h>
#include "assignments.cuh"

typedef unsigned int uint;
typedef struct {
	int32_t lits[3];
} clause3_t;

typedef struct {
	uint32_t litCount, length;
	clause3_t* clauses;
} expression_t;

typedef struct {
	uint8_t device_id;
	char* device_name;
	expression_t* exp;
	clause3_t* cl;
	assignment_t* asmnt;
} device_work_t;


enum sat_result { SAT_SAT, SAT_UNSAT, SAT_ERROR, SAT_UNDETERMINED };
void sort_clauses(expression_t* e);
int read_expression_from_file(FILE*, expression_t**);
void print_expression(expression_t*);
int remap_unsat_lits(expression_t*, assignment_t*);
void swap(expression_t* e, assignment_t* a, int32_t j, int32_t i);
void solver_driver(const expression_t* e, const assignment_t* a, const uint8_t unsetCnt, device_work_t* gpus, uint8_t gpucnt);
__global__ void sub_check_kernel(const expression_t* const, assignment_t* , const uint8_t, const uint8_t , const uint8_t , const uint32_t);
__device__  __inline__ int d_check(const expression_t* const, assignment_t*, const uint64_t);
#endif