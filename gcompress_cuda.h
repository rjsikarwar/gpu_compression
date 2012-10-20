#ifndef __GCOMPRESS_CUDA_H__
#define __GCOMPRESS_CUDA_H__

#include <stdio.h>

#undef BEGIN_C_DECLS
#undef END_C_DECLS
#ifdef __cplusplus
# define BEGIN_C_DECLS extern "C" {
# define END_C_DECLS }
#else
# define BEGIN_C_DECLS /* empty */
# define END_C_DECLS /* empty */
#endif
//====================================================
//CUDA macros
//====================================================
#define THREAD_CONF(grid, block, gridBound, blockBound) do {\
	    block.x = blockBound;\
	    grid.x = gridBound; \
		if (grid.x > 65535) {\
		   grid.x = (int)sqrt((double)grid.x);\
		   grid.y = CEIL(gridBound, grid.x); \
		}\
	}while (0)
#define BLOCK_ID (__umul24(gridDim.y, blockIdx.x) + blockIdx.y)
#define THREAD_ID (threadIdx.x)
#define TID (__umul24(BLOCK_ID, blockDim.x) + THREAD_ID)
#define CEIL(n, d) (n/d + (int)(n%d!=0))

#define ONE_BYTE 0xff
#define TWO_BYTE 0xffff
#define THREE_BYTE 0xffffff
#define FOUR_BYTE 0xffffffff

//====================================================
//Portable data structure
//====================================================
typedef struct
{
	int stream;
	int event;
	int start;
} gcStream_t;

//====================================================
//APIs
//====================================================
BEGIN_C_DECLS
#define INCLUSIVE	0x01
#define EXCLUSIVE	0x02

void test_gpu(int num, int print_num);
int prefixSum( int* d_inArr, int numRecords, int* d_outArr, char flag );
float sumFloat(float* input, int num);
float sumFloat2(float* input, int num, float* output);

//-----------------------------------------------------
//Stream Management
//-----------------------------------------------------
void gc_stream_start(gcStream_t* stream);
void gc_stream_stop(gcStream_t* stream);

//-----------------------------------------------------
//GPU Compression Scheme
//-----------------------------------------------------

//Main Schemes
//
void gc_compress_ns2(gcStream_t stream, int* gpu_ubuf, int entry_num, 
                   short* gpu_cbuf, int max_num);
void gc_decompress_ns2(gcStream_t stream, int* gpu_ubuf, int entry_num, 
                   short* gpu_cbuf, int max_num);


void gc_compress_ns(gcStream_t stream, int* gpu_ubuf, int entry_num, 
                   char* gpu_cbuf, int max_num);
void gc_decompress_ns(gcStream_t stream, int* gpu_ubuf, int entry_num, 
                   char* gpu_cbuf, int max_num);

void gc_compress_nsv(gcStream_t stream, int* gpu_ubuf, int entry_num, 
                   char** gpu_value, char* gpu_len, int* size);
void gc_decompress_nsv(gcStream_t stream, int* gpu_ubuf, int entry_num, 
                   char* gpu_value, char* gpu_len);

void gc_decompress_dict(gcStream_t stream, char* gpu_uncompressed, int entry_num, int* gpu_compressed, char* gpu_dict);

void gc_compress_rle(gcStream_t stream, int* gpu_ubuf, int entry_num,
                     int** gpu_valbuf, int** gpu_lenbuf, int* centry_num);
void gc_decompress_rle(gcStream_t stream, int** gpu_ubuf, int* entry_num, 
                     int* gpu_valbuf, int* gpu_lenbuf, int centry_num);

void gc_compress_bitmap(gcStream_t stream, char* gpu_ubuf, int entry_num,
                        char *gpu_r, char* gpu_n, char* gpu_a);
void gc_decompress_bitmap(gcStream_t stream, char* gpu_ubuf, int entry_num,
                        char *gpu_r, char* gpu_n, char* gpu_a);


//Auxiliary Schemes
void gc_compress_for(gcStream_t stream, int* gpu_ubuf, int entry_num, int* gpu_cbuf, int reference);
void gc_decompress_for(gcStream_t stream, int* gpu_ubuf, int entry_num, int* gpu_cbuf, int reference);

void gc_compress_delta(gcStream_t stream, int* gpu_ubuf, int entry_num, int* gpu_cbuf, int* first_elem);
void gc_decompress_delta(gcStream_t stream, int* gpu_ubuf, int entry_num, int* gpu_cbuf, int first_elem);

void gc_compress_sep(gcStream_t stream, float* gpu_ubuf, int entry_num, int* gpu_lbuf, int* gpu_rbuf);
void gc_decompress_sep(gcStream_t stream, float* gpu_ubuf, int entry_num, int* gpu_lbuf, int* gpu_rbuf);

void gc_compress_scale(gcStream_t stream, float* gpu_ubuf, int entry_num, int* gpu_cbuf);
void gc_decompress_scale(gcStream_t stream, float* gpu_ubuf, int entry_num, int* gpu_cbuf);


//-----------------------------------------------------
//Memory Management
//-----------------------------------------------------
void* gc_malloc(size_t bufsize);
void gc_free(void* gpu_buf);
void* gc_host2device(gcStream_t stream, void* cpu_buf, size_t bufsize);
void* gc_device2host(gcStream_t stream, void* gpu_buf, size_t bufsize);
int byte_num(int max_num);

//-----------------------------------------------------
//query operators
//-----------------------------------------------------
#define LT	0x00
#define LE	0x01
#define GT	0x02
#define GE	0x03
#define NL	0x04
#define NG	0x05
void gc_select_char(gcStream_t stream, char* gpu_column, int entry_num, char low, char op_low, char high, char op_high, char* gpu_pos_vector);
void gc_select_short(gcStream_t stream, short* gpu_column, int entry_num, short low, char op_low, short high, char op_high, char* gpu_pos_vector);
void gc_select_int(gcStream_t stream, int* gpu_column, int entry_num, int low, char op_low, int high, char op_high, char* gpu_pos_vector);
void gc_select_float(gcStream_t stream, float* gpu_column, int entry_num, float low, char op_low, float high, char op_high, char* gpu_pos_vector);
void gc_filter(gcStream_t stream, char* gpu_pos_vector1, char* gpu_pos_vector2, char* gpu_pos_vector3, char* gpu_pos_vector, int entry_num);
void gc_filter_float_value(gcStream_t stream, float* column1, float* column2, int entry_num, char* pos_vector, float* out);
void gc_filter_float_value2(gcStream_t stream, float* column1, char* column2, int entry_num, char* pos_vector, float* out);
void gc_val_vector(gcStream_t stream, int* gpu_column, int entry_num, int max_val, char* gpu_val_vector);
void gc_pos_vector(gcStream_t stream, int* column, char* val_vector, int centry_num, int** pos_list, int* num);
void gc_filter1(gcStream_t stream, int* column, int* pos_list, int entry_num);
void gc_cal_q14(gcStream_t stream, float* price, char* discount, char* type, int* lpartkey_offset_in, 
		int* lpartkey_offset_out, int* lpartkey_len, char* pos_vector, int* lpartkey_pos_list, int* ppartkey_pos_list,
		int entry_num, int centry_num, float* out1, char* out2);
void gc_cal_q14_final(gcStream_t stream, float* out1, char* out2, int entry_num, float sum, float* out);
void gc_intersect(gcStream_t stream, char* pos1, char* pos2, int entry_num, char* pos);

void gc_print_char(char* buf, int num);
void gc_print_int(int* buf, int num);

void gc_scatter(gcStream_t stream, int* column, char* vector, int entry_num, int** out, int* num);
void gc_scatter_float(gcStream_t stream, float* column, char* vector, int entry_num, float** out, int* num);
void gc_sum1(gcStream_t stream, float* price, float* discount, int entry_num);
void gc_sum2(gcStream_t stream, float* price, char* type, int* joinPos, int entry_num);

void gc_compress_ns_long(gcStream_t Stream, long* gpu_ubuf, int entry_num, int* gpu_cbuf, int byteNum) ;
void gc_decompress_ns_long(gcStream_t Stream, long* gpu_ubuf, int entry_num, int* gpu_cbuf, int byteNum);

void gc_compress_nsv_long(gcStream_t Stream, long* gpu_ubuf, int entry_num, char** gpu_value, char* gpu_len, int* size) ;

void test_malloc(int size, int nloop);
void test_int2char();
END_C_DECLS

#endif
