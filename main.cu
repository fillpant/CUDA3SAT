#include <device_functions.h>
#include "device_launch_parameters.h"
#include "satcheck.cuh"
#include "errors.cuh"
#include "assignments.cuh"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>
#include <time.h>

#define THREAD_INDX_1D_GRID_2D_BLOCK (blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)
#define THREAD_INDX_2D_BLOCK (threadIdx.y * blockDim.x + threadIdx.x)

//#define VERBOSE
#define CLAUSE_SORTING
#define ENABLE_DEBUG
//#define BENCHMARK_ROUNDS 10
#ifdef ENABLE_DEBUG
	#define DEBUG(a,...) printf("\n[%10s:%u] " a, __FILE__,__LINE__,__VA_ARGS__)
#else
	#define DEBUG(...) 
#endif

#ifdef __INTELLISENSE__
	#define __syncthreads() printf("you shouldn't be seeing this -- Intelisense compatibility");exit(1);
	#define __syncwarp() printf("you shouldn't be seeing this -- Intelisense compatibility");exit(1);
	#define CUDA_KERNEL(...)
#else
	#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

#define SIGN(a) ((a)<0?-1:1)

#define NOTNULL(a) if(!a) { \
	fprintf(stderr, "Unexpected null pointer at line %u of file %s\n", __LINE__, __FILE__); \
	exit(ERRNO(ERR_NULL_POINTER)); \
}

#define CHECK_CUDA_ERROR(expr) \
{ \
	cudaError_t a__cuda_error = expr; \
	if (a__cuda_error != cudaSuccess) { \
		fprintf(stderr,"ERROR on line %u of file %s: [%s]\n", __LINE__, __FILE__, cudaGetErrorString(a__cuda_error)); \
		exit(a__cuda_error); \
	} \
}


#define MIN_TESTED_COMPUTE_CAPABILITY 5.0
#define MAX_SMEM_BYTES 49152
#define BLOCK_DIM 32
#define BLOCK_CNT 224



static int assgn_dist_score(int32_t,int32_t);
static int memory_distance_comparator(const void*, const void*);
static uint8_t select_compute_devices(device_work_t**, uint8_t);

int main(int argc, char* argv[]) {
	DEBUG("Program starts");
	if (argc == 2) {
		DEBUG("Loading file %s...", argv[1]);
	} else {
		printf("Usage: %s <file.cnf>", argv[0]);
		return 0;
	}

	expression_t* e = NULL;
	int err = read_expression_from_file(fopen(argv[1], "r"), &e);
	if (err) 
		FAIL(err, "Failed to read file: %s\n", argv[1]);

	assignment_t* a = (assignment_t*) malloc(sizeof(assignment_t) * ASSIGNMENT_COUNT(e->litCount));
	NOTNULL(a);

	#ifdef CLAUSE_SORTING
	sort_clauses(e);
	#endif

	#ifdef VERBOSE
	DEBUG("Assignment distance scoring breakdown:");
	for (uint32_t i = 0; i < e->length; ++i) {
		clause3_t* c1 = &e->clauses[i];
		int c1d = assgn_dist_score(c1->lits[0], c1->lits[1]) + assgn_dist_score(c1->lits[1], c1->lits[2]) + assgn_dist_score(c1->lits[0], c1->lits[2]);
		DEBUG("(%3d,%3d,%3d) -> %d,", c1->lits[0], c1->lits[1], c1->lits[2], c1d);
	}
	#endif

	clear_assignment(a, e->litCount);
	
	const int resp = remap_unsat_lits(e, a);
	if (resp <= 0 || resp > 64) {
		FAIL(ERR_UNSUPPORTED, "Too many unsat lits found! Remaping failed.");
	}
	DEBUG("Remapped literals: %u", resp);

	int devices = 0;
	CHECK_CUDA_ERROR(cudaGetDeviceCount(&devices));

	if (devices > UINT8_MAX) {
		printf("WARNING: %d devices detected but only up to %u supported. Ignoring the rest.\n", devices, UINT8_MAX);
		devices = UINT8_MAX;
	}

	device_work_t* dev_confs;
	devices = select_compute_devices(&dev_confs, (uint8_t) devices);

	if (!devices) 
		FAIL(ERR_NO_DEVICE, "No compatible compute enabled CUDA device found!");

	DEBUG("Using %u devices...", devices);
	solver_driver(e, a, (uint8_t) resp, dev_confs, (uint8_t) devices);

	DEBUG("Completed. Freeing memory.");
	for (uint8_t i = 0; i < devices; i++)
		free(dev_confs[i].device_name);
	free(dev_confs);
	free(a);
	free(e -> clauses);
	free(e);
	DEBUG("Success");
	return EXIT_SUCCESS;
}

static uint8_t select_compute_devices(device_work_t** confs, uint8_t devices) {
	device_work_t* dev_confs = (device_work_t*) malloc(sizeof(device_work_t) * devices);
	memset(dev_confs, 0, sizeof(device_work_t) * devices);
	uint8_t cnt = 0; 
	cudaDeviceProp prop;
	for (uint8_t i = 0; i < devices; ++i) {
		CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));
		DEBUG("Check device %u (%s)", i, prop.name);
		if (prop.major >= MIN_TESTED_COMPUTE_CAPABILITY && prop.sharedMemPerBlock >= MAX_SMEM_BYTES) {
			DEBUG("CC Major: %d, %d SMs, %d threads/block, peak mem %dkhz (%lluB SM)\n",
										prop.major,
										prop.multiProcessorCount,
										prop.maxThreadsPerBlock,
										prop.memoryClockRate,
										prop.sharedMemPerBlock);
			dev_confs[cnt].device_name = strdup(prop.name);
			dev_confs[cnt++].device_id = i;
		} else {
			DEBUG("Device incompatible/unsupported.\n");
		}
	}
	dev_confs = (device_work_t*) realloc(dev_confs, sizeof(device_work_t) * cnt);
	*confs = dev_confs;
	return cnt;
}


inline void sort_clauses(expression_t* e) {
	qsort(e->clauses, e->length, sizeof(clause3_t), memory_distance_comparator);
}

void swap(expression_t* e, assignment_t* a, int32_t j, int32_t i) {
	if (j == i)
		return;

	for (uint32_t o = 0; o < e->length; ++o) {
		int32_t* cl = e->clauses[o].lits;

		for (uint32_t x = 0; x < 3; ++x) {
			if (cl[x] == j || cl[x] == -j)
				cl[x] = SIGN(cl[x]) * i;
			else if (cl[x] == i || cl[x] == -i)
				cl[x] = SIGN(cl[x]) * j;
		}
	}
	litval_t s = get_lit(a, j);
	set_lit(a, j, get_lit(a, i));
	set_lit(a, i, s);
}


int remap_unsat_lits(expression_t* e, assignment_t* a) {
	uint32_t i, j;
	for (i = 0, j = 0; j <= e->litCount && i <= 64; ++j) {
		if (j > i && get_lit(a, j) == LIT_UNSET) {
			++i;
			swap(e, a, j, i);
			j = 0;
		}
	}
	assert(i <= 64);
	return i;

}

void print_expression(expression_t* e) {
	printf("p cnf %d %d\n", e->litCount, e->length);
	for (unsigned i = 0; i < e->length; ++i)
		printf("%d %d %d\n", e->clauses[i].lits[0], e->clauses[i].lits[1], e->clauses[i].lits[2]);
	printf("\n\n");
}

__global__ void print_expression_kernel(const expression_t* const g_e) {
	printf("Expression of length %u, with %u lits:\n", g_e->length, g_e->litCount);
	for (uint32_t i = 0; i < g_e->length; ++i) {
		printf("(%d,%d,%d)\n", g_e->clauses[i].lits[0], g_e->clauses[i].lits[1], g_e->clauses[i].lits[2]);
	}
}

void solver_driver(const expression_t* e, const assignment_t* a, const uint8_t unsetCnt, device_work_t* gpus, uint8_t gpucnt) {
	for (uint8_t i = 0; i < gpucnt; ++i) {
		uint8_t device_id = gpus[i].device_id;
		DEBUG("Setting up device %u..", device_id);
		CHECK_CUDA_ERROR(cudaSetDevice(device_id));
		CHECK_CUDA_ERROR(cudaDeviceReset());
		CHECK_CUDA_ERROR(cudaFuncSetCacheConfig(sub_check_kernel, cudaFuncCachePreferL1));
		
		expression_t* d_e = NULL;
		clause3_t* d_e_c = NULL;
		assignment_t* d_a = NULL;
		DEBUG("Allocating memory for the expression on GPU #%u", device_id);
		CHECK_CUDA_ERROR(cudaMalloc(&d_e, sizeof(expression_t)));
		DEBUG("Allocating memory for the clauses on GPU #%u", device_id);
		CHECK_CUDA_ERROR(cudaMalloc(&d_e_c, sizeof(clause3_t) * e->length));
		DEBUG("Allocating memory for the assignment on GPU #%u", device_id);
		CHECK_CUDA_ERROR(cudaMalloc(&d_a, sizeof(assignment_t) * ASSIGNMENT_COUNT(e->litCount)));

		DEBUG("Copy expression to GPU #%u", device_id);
		CHECK_CUDA_ERROR(cudaMemcpy(d_e, e, sizeof(expression_t), cudaMemcpyHostToDevice));
		DEBUG("Copy clauses to GPU #%u", device_id);
		CHECK_CUDA_ERROR(cudaMemcpyAsync(d_e_c, e->clauses, sizeof(clause3_t) * e->length, cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaMemcpyAsync(&d_e->clauses, &d_e_c, sizeof(clause3_t*), cudaMemcpyHostToDevice));
		DEBUG("Copy assignment to GPU #%u", device_id);
		CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a, a, sizeof(assignment_t) * ASSIGNMENT_COUNT(e->litCount), cudaMemcpyHostToDevice));

		gpus[i].exp = d_e;
		gpus[i].asmnt = d_a;
		gpus[i].cl = d_e_c;
		DEBUG("GPU #%u set up\n", device_id);
	}
	//########### Launch ###########

	const size_t smem_size = sizeof(assignment_t) * ASSIGNMENT_COUNT(e->litCount);
	assert(smem_size < MAX_SMEM_BYTES);
	assert(BLOCK_DIM*BLOCK_DIM >= ASSIGNMENT_COUNT(e->litCount));//Can't copy assignment with current logic if this holds true.... 
	DEBUG("%llu bytes of shared memory needed", smem_size);

	const uint64_t per_device_solvers = BLOCK_DIM * BLOCK_DIM * BLOCK_CNT;
	DEBUG("%llu solvers/GPU (%u blocks of dim %u)", per_device_solvers, BLOCK_CNT, BLOCK_DIM);

	const dim3 block(BLOCK_DIM, BLOCK_DIM);
	DEBUG("Limit/solver: %u\n", (uint32_t) ceil((1LLU << unsetCnt) / ((double)(per_device_solvers))*gpucnt));

#ifdef BENCHMARK_ROUNDS
	for (uint bm = 0; bm < BENCHMARK_ROUNDS; ++bm) {
#endif

	for (uint8_t i = 0; i < gpucnt; ++i) {
		DEBUG("Kernel Launch on GPU #%u", gpus[i].device_id);
		CHECK_CUDA_ERROR(cudaSetDevice(gpus[i].device_id));
		sub_check_kernel CUDA_KERNEL(BLOCK_CNT, block, smem_size)(gpus[i].exp, gpus[i].asmnt, unsetCnt, gpus[i].device_id, gpucnt, per_device_solvers);
	}
	//########## Wait for all devices ###########
	for (uint8_t i = 0; i < gpucnt; ++i) {
		DEBUG("Synchronize with GPU #%u...", gpus[i].device_id);
		CHECK_CUDA_ERROR(cudaSetDevice(gpus[i].device_id));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	}
	DEBUG("All kernels exited.\n");
#ifdef BENCHMARK_ROUNDS 
	}
#endif

	//########## Free up memory ##########
	for (uint i = 0; i < gpucnt; ++i) {
		CHECK_CUDA_ERROR(cudaSetDevice(gpus[i].device_id));
		DEBUG("Releasing memory on GPU #%u", gpus[i].device_id);
		CHECK_CUDA_ERROR(cudaFree(gpus[i].exp));
		CHECK_CUDA_ERROR(cudaFree(gpus[i].cl));
		CHECK_CUDA_ERROR(cudaFree(gpus[i].asmnt));
	}
	DEBUG("Driver done");
}



__global__ void sub_check_kernel(const expression_t* const __restrict__ g_e, assignment_t* __restrict__ g_a, const uint8_t unsetCnt, 
																			const uint8_t device_id, const uint8_t devices, const uint32_t solvers) {
	extern __shared__ assignment_t s_assgn[];
	// Copy assignment to shared memory
	if (THREAD_INDX_2D_BLOCK < ASSIGNMENT_COUNT(g_e->litCount))
		s_assgn[THREAD_INDX_2D_BLOCK] = g_a[THREAD_INDX_2D_BLOCK];
	
	// Calculate per thread limit
	const uint64_t limit = (uint64_t) ceil((1LLU << unsetCnt) / (((double)solvers)*devices));
	uint64_t i, local_assignmen = (solvers*device_id) + THREAD_INDX_1D_GRID_2D_BLOCK;
	for (i = 0; i < limit; ++i) {
		// Branch nessesary to prevent OOM access
		if (local_assignmen > (1LLU << unsetCnt)) 
			break;
		const int check = d_check(g_e, s_assgn, local_assignmen);
		__syncwarp(); //Sync after check which may cause divergence
		if (check) {
			printf("SAT!: %llux\n", local_assignmen);
			break;
		}
		local_assignmen += solvers;
	}
}

__device__ __forceinline__ int d_check(const expression_t* const __restrict__ g_e, assignment_t* const __restrict__ g_a, const uint64_t assgn) {
	register bool whole = 1;
	const uint32_t len = g_e->length;
	for (uint32_t i = 0; whole && (i < len); ++i) {
		const int32_t* __restrict__ const clause = g_e->clauses[i].lits;
		const int32_t a = clause[0], b = clause[1], c = clause[2];
		whole &= ((is_lit_sated_complete(g_a, a, assgn)) | (is_lit_sated_complete(g_a, b, assgn)) | (is_lit_sated_complete(g_a, c, assgn)));
	}
	return whole;
}


int getLines(char*** arr, unsigned long long* r_lineCount, FILE* f) {
	if (!f || feof(f) || ferror(f)) {
		return ERRNO(ERR_CANNOT_OPEN_FILE);
	}
	fseek(f, 0L, SEEK_END);
	size_t fileSize = ((size_t) ftell(f)) + 1L;//+1, ftel is 0 indexed.
	rewind(f);

	char* in = (char*) malloc(sizeof(char) * fileSize + 3);
	memset(in, 0, sizeof(char) * fileSize + 3);
	if (!in) {
		return ERRNO(ERR_NOT_ENOUGH_MEMORY);
	}

	fread(in, sizeof(char), fileSize, f);
	in[fileSize - 1] = '\n';
	in[fileSize] = '\0';


	unsigned long long lineCount = 0;//should be 1 but we add an artificial \n at the end of the file so all good.
	for (unsigned long long i = 0; i < fileSize; i++)
		if (in[i] == '\n')
			lineCount++;
	*r_lineCount = lineCount;

	char** l_arr = (char**) malloc(sizeof(char*) * lineCount);
	if (!l_arr) {
		return ERRNO(ERR_NOT_ENOUGH_MEMORY);
	}
	unsigned long long lines = 0;
	char* strt = in;
	char* nl = NULL;
	while ((nl = strchr(strt, '\n'))) {
		//nl will always be strt+diff
		size_t diff = (nl - strt) / sizeof(char);
		if (diff == 0) {
			l_arr[lines] = NULL;
		} else {
			l_arr[lines] = (char*) malloc(sizeof(char) * (diff + 1));
			if (!l_arr[lines]) {
				return ERRNO(ERR_NOT_ENOUGH_MEMORY);
			}

			memcpy(l_arr[lines], strt, diff);
			l_arr[lines][diff] = '\0';
		}
		++lines;
		strt = nl + 1;
	}
	*arr = l_arr;
	free(in);
	return 0;
}

int read_expression_from_file(FILE* f, expression_t** putWhere) {
	char** lines_arr = NULL;
	uint64_t lineCount;
	uint32_t vars = 0, clausec = 0;
	expression_t* e = NULL;

	int err = getLines(&lines_arr, &lineCount, f);
	if (err) 
		goto done;

	e = (expression_t*) malloc(sizeof(expression_t));
	for (uint32_t cl = 0, li = 0; li < lineCount; ++li) {
		char* line = lines_arr[li];
		if (!line)
			continue; //Skip blank line
		//skip spaces, tabs at start 
		for (; line[0] == ' ' || line[0] == '\t'; ++line);
		//skip comment
		if (line[0] == 'c')
			continue;
		else if (line[0] == '%')
			break;
		else if (line[0] == 'p') {
			if (vars == 0 && clausec == 0) {
				int match = sscanf(line, "p cnf %lu %lu", &vars, &clausec);
				if (match != 2) { 
					err = ERRNO(ERR_ILLEGAL_LINE);	
					goto done; 
				}
				e->clauses = (clause3_t*) malloc(sizeof(clause3_t) * clausec);
				e->length = clausec;
				e->litCount = vars;
			} else { 
				err = ERRNO(ERR_DUPLICATE_PROBLEM_STATEMENT);
				goto done; 
			}

		} else {
			if (vars == 0 || clausec == 0) {
				err = ERRNO(ERR_UNEXPECTED_EMPTY_LINE); 
				goto done;
			} else {
				int32_t a, b, c;
				int match = sscanf(line, "%d %d %d 0", &a, &b, &c);
				#ifdef VERBOSE
				DEBUG("Matched %llu/3 for line %llu", match, li)
				#endif
				if (match != 3) {
					DEBUG("Failed to parse line %u", li);
					err = ERRNO(ERR_ILLEGAL_LINE);
					goto done;
				}

				if (cl >= clausec) {
					err = ERRNO(ERR_ILLEGAL_CLAUSE_COUNT); 
					goto done; 
				}

				if (a == 0 || b == 0 || c == 0) { 
					err = ERRNO(ERR_ILLEGAL_VARIABLE); 
					goto done; 
				}

				e->clauses[cl].lits[0] = a;
				e->clauses[cl].lits[1] = b;
				e->clauses[cl].lits[2] = c;
				++cl;
			}
		}
	}
done:
	if (lines_arr) {
		for (uint64_t i = 0; i < lineCount; ++i) {
			if (lines_arr[i])
				free(lines_arr[i]);
		}
		free(lines_arr);
	}

	if (err) {
		if (e) {
			if (e->clauses) {
				free(e->clauses);
			}
			free(e);
		}
	} else {
		*putWhere = e;
	}
	return err;
}



//*************STATIC HELPERS**************
static int assgn_dist_score(int32_t one, int32_t two) {
	one = abs(one);
	two = abs(two);

	if (!one || !two) 
		FAIL(ERRNO(ERR_UNSUPPORTED),"");

	const int32_t assgnWordIndxOne = ((one - 1) / LITS_PER_WORD);
	const int32_t assgnWordIndxTwo = ((two - 1) / LITS_PER_WORD);
	const uint32_t dist = abs(assgnWordIndxOne - assgnWordIndxTwo);
	if (dist <= 5) 
		return 5-dist;//4-dist means the closer to 4 distance the worse the returned score. dist 5 = score 0.
	if (dist == 0) 
		return 10;//Very good, the two are close together.
	return 5-dist;//The further appart (out of the 4 distance box) they are, the worse the score. (negative)

}

static int memory_distance_comparator(const void* a, const void* b) {
	clause3_t* c1 = (clause3_t*)a;
	clause3_t* c2 = (clause3_t*)b;
	int c1d = assgn_dist_score(c1->lits[0], c1->lits[1]) + assgn_dist_score(c1->lits[1], c1->lits[2]) + assgn_dist_score(c1->lits[0], c1->lits[2]);
	int c2d = assgn_dist_score(c2->lits[0], c2->lits[1]) + assgn_dist_score(c2->lits[1], c2->lits[2]) + assgn_dist_score(c2->lits[0], c2->lits[2]);
	return (c1d - c2d);
}
