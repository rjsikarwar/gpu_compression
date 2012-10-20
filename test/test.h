#include "gcompress_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int* 
genDataIntSC(int num, int c)
{
	int* buf = (int*)malloc(sizeof(int)*num);

	int run1 = num / c;
	int run2 = num % c;

	int i;
	for (i = 0; i < c; ++i)
	{
		for (int j = 0; j < run1; j++)
			buf[i*run1 + j] = i;
	}

	for (int j = 0; j < run2; j++)
		buf[i*run1 + j] = i;
	return buf;
}


static int* 
genDataIntS(int num)
{
	int* buf = (int*)malloc(sizeof(int)*num);
	for (int i = 0; i < num; ++i)
		buf[i] = i;
	return buf;
}
static int* 
genDataIntR(int num, int min, int max)
{
	if (min > max)
	{
		return NULL;
	}
	srand(time(0));
	int* buf = (int*)malloc(sizeof(int)*num);
	for (int i = 0; i < num; ++i)
	{
		int val = 0;
		while (val < min) val = rand() % max;
		buf[i] = val;
	}
	return buf;
}

static void
printChar(char* input, int num, int printNum)
{
	if (num < printNum) return;

	for (int i = 0; i < printNum; ++i)
	{
		printf("%d: %d , ", i, input[i]);
	}
	printf("\n");
}


static void
printInt(int* input, int num, int printNum)
{
	if (num < printNum) return;

	for (int i = 0; i < printNum; ++i)
	{
		printf("%d: %d , ", i, input[i]);
	}
	printf("\n");
}

void testRLE(int numEntry)
{
	int numCentry = 5000;
	int* cpu_uncompressed = genDataIntSC(numEntry, 5000);
//	printInt(cpu_uncompressed, numEntry, numEntry);
	int* gpu_uncompressed = NULL;
	int* gpu_compressed = NULL;
	int* gpu_compressed_len = NULL;
	int* gpu_raw = NULL;

	//-----------------------------------------------------
	//RLE
	//-----------------------------------------------------
	gcStream_t s;
	gc_stream_start(&s);
	gpu_uncompressed = gc_host2device(s, cpu_uncompressed, numEntry * sizeof(int));
	gpu_compressed = (int*)gc_malloc(numCentry * sizeof(int));
	gpu_compressed_len = (int*)gc_malloc(numCentry * sizeof(int));
	gc_compress_rle(s, gpu_uncompressed, numEntry, gpu_compressed, gpu_compressed_len, numCentry);
	gc_stream_stop(&s);
	gc_free(gpu_uncompressed);

	//printInt(gpu_compressed, numEntry, numEntry);
	gc_stream_start(&s);
	gpu_raw = (int*)gc_malloc(CEIL(numEntry, 4) * 4 * sizeof(int));
	gc_decompress_rle(s, gpu_raw, numEntry, gpu_compressed, gpu_compressed_len, numCentry);
	int* h_uncompressed = gc_device2host(s, gpu_raw, numEntry*sizeof(int));
	printInt(h_uncompressed, numEntry, 10);
	//int* h_uncompressed = gc_device2host(s, gpu_raw, CEIL(numEntry, 4)*4*sizeof(int));
	//printInt(h_uncompressed, numEntry, numEntry);
	//free(h_uncompressed);
	gc_stream_stop(&s);
	//printInt(gpu_raw, numEntry, numEntry);

	gc_free(gpu_compressed);
	gc_free(gpu_compressed_len);
	gc_free(gpu_raw);
	free(cpu_uncompressed);
}


void testNS()
{
	int numEntry = 100;
	int* cpu_uncompressed = genDataIntS(numEntry);
//	printInt(cpu_uncompressed, numEntry, numEntry);
	int* gpu_uncompressed = NULL;
	char* gpu_compressed = NULL;
	int* gpu_raw = NULL;

	//-----------------------------------------------------
	//NS
	//-----------------------------------------------------
	gcStream_t s;
	gc_stream_start(&s);
	gpu_uncompressed = gc_host2device(s, cpu_uncompressed, numEntry * sizeof(int));
	gpu_compressed = (char*)gc_malloc(numEntry * sizeof(char));
	int max = numEntry-1;
	int byteNum = byte_num(max);
	gc_compress_ns(s, gpu_uncompressed, numEntry, gpu_compressed, byteNum);
	gc_stream_stop(&s);
	//printInt(cpu_uncompressed, numEntry, numEntry);
	//printChar(gpu_compressed, numEntry, numEntry);
	gc_stream_start(&s);
	gpu_raw = (int*)gc_malloc(numEntry * sizeof(int));
	gc_decompress_ns(s, gpu_raw, numEntry, gpu_compressed, byteNum);
	gc_stream_stop(&s);
	//printChar(gpu_compressed, numEntry, numEntry);
	//printInt(gpu_uncompressed, numEntry, numEntry);
	//printInt(gpu_raw, numEntry, numEntry);
	gc_free(gpu_uncompressed);
	gc_free(gpu_compressed);
	gc_free(gpu_raw);
	free(cpu_uncompressed);
}


void testDelta()
{
	int numEntry = 100;
	int* cpu_uncompressed = genDataIntS(numEntry);
//	printInt(cpu_uncompressed, numEntry, numEntry);
	int* gpu_uncompressed = NULL;
	int* gpu_compressed = NULL;
	int* gpu_raw = NULL;

	//-----------------------------------------------------
	//Delta
	//-----------------------------------------------------
	gcStream_t s;
	gc_stream_start(&s);
	gpu_uncompressed = gc_host2device(s, cpu_uncompressed, numEntry * sizeof(int));
	gpu_compressed = (int*)gc_malloc(numEntry * sizeof(int));
	int first = 0;
	gc_compress_delta(s, gpu_uncompressed, numEntry, gpu_compressed, &first);
	gc_stream_stop(&s);
//	printInt(gpu_compressed, numEntry, numEntry);
	gc_stream_start(&s);
	gpu_raw = (int*)gc_malloc(numEntry * sizeof(int));
	gc_decompress_delta(s, gpu_raw, numEntry, gpu_compressed, first);

	gc_stream_stop(&s);
	//printInt(gpu_compressed, numEntry, numEntry);
	//printInt(gpu_uncompressed, numEntry, numEntry);
	//printInt(gpu_raw, numEntry, numEntry);
	gc_free(gpu_uncompressed);
	gc_free(gpu_compressed);
	gc_free(gpu_raw);
	free(cpu_uncompressed);
}

void testFor()
{
	int numEntry = 10500000;
	int min = 100;
	int max = 125;
	int* cpu_uncompressed = genDataIntR(numEntry, min, max);
	//printInt(cpu_uncompressed, numEntry, numEntry);
	int* gpu_uncompressed = NULL;
	int* gpu_compressed = NULL;
	int* gpu_raw = NULL;

	//-----------------------------------------------------
	//FOR
	//-----------------------------------------------------
	int ref = 100;
	gcStream_t s;
	gc_stream_start(&s);
	gpu_uncompressed = gc_host2device(s, cpu_uncompressed, numEntry * sizeof(int));
	gpu_compressed = (int*)gc_malloc(numEntry * sizeof(int));

	unsigned timer;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
	gc_compress_for(s, gpu_uncompressed, numEntry, gpu_compressed, ref);
	gc_stream_stop(&s);
	cutStopTimer(timer);
	double ftime = cutGetTimerValue(timer);
	printf("%f ms, %f MB/s\n", ftime, numEntry * sizeof(int) / ftime);

	gc_free(gpu_uncompressed);

	gc_stream_start(&s);
	gpu_raw = (int*)gc_malloc(numEntry * sizeof(int));
	gc_decompress_for(s, gpu_raw, numEntry, gpu_compressed, ref);
	int* h_uncompressed = gc_device2host(s, gpu_raw, numEntry*sizeof(int));
	printInt(h_uncompressed, numEntry, 10);
	
	gc_stream_stop(&s);
	//printInt(gpu_compressed, numEntry, 10);
	//printInt(gpu_uncompressed, numEntry, 10);
	//printInt(gpu_raw, numEntry, 10);
	gc_free(gpu_compressed);
	gc_free(gpu_raw);
	free(cpu_uncompressed);
}

//-------------------------------------------
//Test suites
//-------------------------------------------
void testStream()
{
	gcStream_t stream1;
	gcStream_t stream2;

	gc_stream_start(&stream1);
	gc_stream_start(&stream2);

	int bufsize = 1024*1024*100;

	void* test_cpu_data = malloc(bufsize);
	void* test_gpu_data1 = gc_host2device(stream1, test_cpu_data, bufsize);
	void* test_gpu_data2 = gc_host2device(stream2, test_cpu_data, bufsize);

	void* test_cpu_data1 = gc_device2host(stream1, test_gpu_data1, bufsize);
	void* test_cpu_data2 = gc_device2host(stream2, test_gpu_data2, bufsize);

	test_gpu(100, 20);

	gc_stream_stop(&stream1);
	gc_stream_stop(&stream2);

	free(test_cpu_data);
	free(test_cpu_data1);
	free(test_cpu_data2);
	gc_free(test_gpu_data1);
	gc_free(test_gpu_data2);
}


