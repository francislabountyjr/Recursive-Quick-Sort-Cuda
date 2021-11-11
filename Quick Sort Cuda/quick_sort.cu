#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <helper_cuda.h>
#include <helper_string.h>

#define MAX_DEPTH 24
#define INSERTION_SORT 32

// use selection sort when data reaches the max depth level
__device__ void selection_sort(unsigned int* data, int left, int right)
{
	for (int i = left; i <= right; ++i)
	{
		unsigned min_val = data[i];
		int min_idx = i;

		// find the smallest value in the range [left, right]
		for (int j = i + 1; j <= right; ++j)
		{
			unsigned val_j = data[j];
			if (val_j < min_val)
			{
				min_idx = j;
				min_val = val_j;
			}
		}

		// swap the values
		if (i != min_idx)
		{
			data[min_idx] = data[i];
			data[i] = min_val;
		}
	}
}

// quicksort algorithm using dynamic parallelism sorting recursively until the max depth is reached
__global__ void cdp_simple_quicksort(unsigned int* data, int left, int right, int depth)
{
	if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT)
	{
		selection_sort(data, left, right);
		return;
	}

	unsigned int* lptr = data + left;
	unsigned int* rptr = data + right;
	unsigned int pivot = data[(left + right) / 2];

	// partitioning
	while (lptr <= rptr)
	{
		// find the next left and right hand values to swap
		unsigned int lval = *lptr;
		unsigned int rval = *rptr;

		// move the left pointer as long as the pointed element is less than the pivot
		while (lval < pivot)
		{
			lptr++;
			lval = *lptr;
		}

		// move the right pointer as long as the pointed element is larger than the pivot
		while (rval > pivot)
		{
			rptr--;
			rval = *rptr;
		}

		// if the points are valid, conduct the swap
		if (lptr <= rptr)
		{
			*lptr++ = rval;
			*rptr-- = lval;
		}
	}

	// recursive set up
	int nright = rptr - data;
	int nleft = lptr - data;

	// launch a new block to sort the left part
	if (left < (rptr - data))
	{
		cudaStream_t s;
		cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
		cdp_simple_quicksort<<<1, 1, 0, s>>>(data, left, nright, depth + 1);
		cudaStreamDestroy(s);
	}

	// launch a new block to sort the right part
	if ((lptr - data) < right)
	{
		cudaStream_t s1;
		cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
		cdp_simple_quicksort<<<1, 1, 0, s1>>>(data, nleft, right, depth + 1);
		cudaStreamDestroy(s1);
	}
}

// call quicksort kernel from the host
void run_qsort(unsigned int* data, unsigned int nitems)
{
	// prepare CDP for the max depth 'MAX_DEPTH'
	checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

	// launch kernel on device
	int left = 0;
	int right = nitems - 1;
	printf("Launching kernel on the GPU\n");
	cdp_simple_quicksort<<<1, 1>>>(data, left, right, 0);
	checkCudaErrors(cudaDeviceSynchronize());
}

// initialize data on host
void initialize_data(unsigned int* dst, unsigned int nitems, int seed)
{
	srand(seed);

	// fill dst with random values
	for (unsigned i = 0; i < nitems; i++)
	{
		dst[i] = rand() % nitems;
	}
}

// verify the results
void check_results(int n, unsigned int* results_d)
{
	unsigned int* results_h = new unsigned[n];
	checkCudaErrors(cudaMemcpy(results_h, results_d, n * sizeof(unsigned), cudaMemcpyDeviceToHost));

	for (int i = 1; i < n; i++)
	{
		if (results_h[i - 1] > results_h[i])
		{
			printf("Invalid item [%d]: %d greater than %d\n", i - 1, results_h[i - 1], results_h[i]);
			exit(EXIT_FAILURE);
		}
	}

	printf("OK\n");
	delete[] results_h;
}

int main()
{
	int num_items = 2048;

	// create input data
	unsigned int* h_data = 0;
	unsigned int* d_data = 0;

	// allocate CPU memory and initialize data
	h_data = (unsigned int*)malloc(num_items * sizeof(unsigned int));
	initialize_data(h_data, num_items, 2021);

	// allocate GPU memory
	checkCudaErrors(cudaMalloc((void**)&d_data, num_items * sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpy(d_data, h_data, num_items * sizeof(unsigned int), cudaMemcpyHostToDevice));

	// execute
	printf("Running quicksort on %d elements\n", num_items);
	run_qsort(d_data, num_items);

	// check the result
	printf("Validating results: ");
	check_results(num_items, d_data);

	// cleanup
	checkCudaErrors(cudaFree(d_data));
	free(h_data);
}