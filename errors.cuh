#ifndef ERRORS_CUH
#define ERRORS_CUH
	#define ERR_TOO_MANY_DEVICES 5
	#define ERR_NO_DEVICE 4
	#define ERR_ILLEGAL_LINE 3
	#define ERR_UNEXPECTED_EMPTY_LINE 2
	#define ERR_NOT_ENOUGH_MEMORY 1
	#define ERR_CANNOT_OPEN_FILE -1
	#define ERR_ILLEGAL_CLAUSE_COUNT -2
	#define ERR_DUPLICATE_PROBLEM_STATEMENT -3
	#define ERR_ILLEGAL_VARIABLE -4
	#define ERR_NULL_POINTER -5
	#define ERR_UNSUPPORTED -6
	
	extern int errno;

	#ifdef __CUDACC__
		#define ERRNO(num) (num)
	#else
		#define ERRNO(num) (errno=(num))
	#endif

	#ifdef __CUDACC__
		#define FAIL(code,text,...) {printf("\nERR %u [%s:%u]: " text, code, __FILE__, __LINE__, __VA_ARGS__);exit(code);}
	#else
		#define FAIL(code,text,...) {fprintf(stderr, "\nERR %u [%s:%u]: " text, __FILE__, __LINE__, (code), __VA_ARGS__);exit(code);}
	#endif

#endif