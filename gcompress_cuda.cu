#include <stdio.h>
#include <cutil_inline.h>
#include <cudpp/cudpp.h>
#include "gcompress_cuda.h"
//#define NS3

int byte_num(int max_num)
{
	if (max_num > THREE_BYTE) return 4;
	if (max_num > ONE_BYTE) return 2;
	return 1;
}

void gc_print_int(int* buf, int num)
{
	int* cpu_val = (int*)malloc(num*sizeof(int));
        cudaMemcpy(cpu_val, buf, num*sizeof(int), cudaMemcpyDeviceToHost);
        //cudaMemcpy(cpu_val, lpartkeyVal, maxValue, cudaMemcpyDeviceToHost);
        for (int i = 0; i < num; i++)
                printf("%d\n", cpu_val[i]);
        free(cpu_val);


}

void gc_print_char(char* buf, int num)
{
	char* cpu_val = (char*)malloc(num);
        cudaMemcpy(cpu_val, buf, num, cudaMemcpyDeviceToHost);
        //cudaMemcpy(cpu_val, lpartkeyVal, maxValue, cudaMemcpyDeviceToHost);
        for (int i = 0; i < num; i++)
                printf("%d\n", cpu_val[i]);
        free(cpu_val);


}
//=============================================================================
//query operators
//=============================================================================
__global__ void gc_sum1_kernel(float* price, float* discount, int entry_num)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;
	
	float4* out = (float4*)price;
	float4* raw = (float4*)discount;

	float4 r = raw[ttid];
	float4 o = out[ttid];

	o.x = o.x * (1.0f - r.x);
	o.y = o.y * (1.0f - r.y);
	o.z = o.z * (1.0f - r.z);
	o.w = o.w * (1.0f - r.w);

	out[ttid] = o;
}

void gc_sum1(gcStream_t stream, float* price, float* discount, int entry_num)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	gc_sum1_kernel<<<gridDim, blockDim, stream.stream>>>(price, discount, entry_num);
	CUT_CHECK_ERROR("gc_intersect");
}

__global__ void gc_sum2_kernel(float* price, char* type, int* joinPos, int entry_num)
{
	int ttid = TID;
	if (ttid >= entry_num) return;
	
	if (type[joinPos[ttid]] < 125) price[ttid] = 0.0f;
}

void gc_sum2(gcStream_t stream, float* price, char* type, int* joinPos, int entry_num)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(entry_num, blockDim.x), blockDim.x);
	gc_sum2_kernel<<<gridDim, blockDim, stream.stream>>>(price, type, joinPos, entry_num);
	CUT_CHECK_ERROR("gc_intersect");

}

__global__ void gc_scatter_hist_kernel(char* vector, int entry_num, int* hist)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 8)) return;
	
	int index = ttid + ttid;
	char c = vector[ttid];
	int4* out = (int4*)hist;
	int4 o;

	if (c & 0x80) o.x = 1; else o.x = 0;
	if (c & 0x40) o.y = 1; else o.y = 0;
	if (c & 0x20) o.z = 1; else o.z = 0;
	if (c & 0x10) o.w = 1; else o.w = 0;

	out[index] = o;

	if (c & 0x08) o.x = 1; else o.x = 0;
	if (c & 0x04) o.y = 1; else o.y = 0;
	if (c & 0x02) o.z = 1; else o.z = 0;
	if (c & 0x01) o.w = 1; else o.w = 0;

	out[index+1] = o;

}

__global__ void gc_scatter_float_kernel(float* column, int* offset, int* hist, int entry_num, float* out)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;
	
	int4* h4 = (int4*)hist;
	int4* o4 = (int4*)offset;
	float4* c4 = (float4*)column;

	int4 h = h4[ttid];
	int4 o = o4[ttid];
	float4 c = c4[ttid];

	if (h.x == 1) out[o.x] = c.x;
	if (h.y == 1) out[o.y] = c.y;
	if (h.z == 1) out[o.z] = c.z;
	if (h.w == 1) out[o.w] = c.w;
}

void gc_scatter_float(gcStream_t stream, float* column, char* vector, int entry_num, float** out, int* num)
{
	int* hist = (int*)gc_malloc(sizeof(int) * entry_num);

	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 8), blockDim.x), blockDim.x);
	gc_scatter_hist_kernel<<<gridDim, blockDim, stream.stream>>>(vector, entry_num, hist);
	CUT_CHECK_ERROR("gc_intersect");

	int* offset = (int*)gc_malloc(sizeof(int)*entry_num);
	*num = prefixSum(hist, entry_num, offset, EXCLUSIVE);

	*out = (float*)gc_malloc(*num * sizeof(float));
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	gc_scatter_float_kernel<<<gridDim,blockDim, stream.stream>>>(column, offset, hist, entry_num, *out);
	CUT_CHECK_ERROR("gc_intersect");
	gc_free(offset);
	gc_free(hist);
}

__global__ void gc_scatter_kernel(int* column, int* offset,int entry_num, int* out)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;
	
	int4* o4 = (int4*)offset;
	int4* c4 = (int4*)column;

	int4 o = o4[ttid];
	int4 c = c4[ttid];

	if (o.x != -1) out[o.x] = c.x;
	if (o.y != -1) out[o.y] = c.y;
	if (o.z != -1) out[o.z] = c.z;
	if (o.w != -1) out[o.w] = c.w;
}
__global__ void gc_scatter_fix_kernel(int* offset, int entry_num, int* hist)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;
	
	int4* h4 = (int4*)hist;
	int4* o4 = (int4*)offset;

	int4 h = h4[ttid];
	int4 o = o4[ttid];

	if (h.x == 0) o.x = -1;
	if (h.y == 0) o.y = -1;
	if (h.z == 0) o.z = -1;
	if (h.w == 0) o.w = -1;

	o4[ttid] = o;
}

 
void gc_scatter(gcStream_t stream, int* column, char* vector, int entry_num, int** out, int* num)
{
	int* hist = (int*)gc_malloc(sizeof(int) * entry_num);

	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 8), blockDim.x), blockDim.x);
	gc_scatter_hist_kernel<<<gridDim, blockDim, stream.stream>>>(vector, entry_num, hist);
	CUT_CHECK_ERROR("gc_intersect");

	int* offset = (int*)gc_malloc(sizeof(int)*entry_num);
	*num = prefixSum(hist, entry_num, offset, EXCLUSIVE);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	gc_scatter_fix_kernel<<<gridDim, blockDim, stream.stream>>>(offset, entry_num, hist);
	CUT_CHECK_ERROR("gc_intersect");
	gc_free(hist);

	printf("%d\n", *num);
	*out = (int*)gc_malloc(*num * sizeof(int));
	gc_scatter_kernel<<<gridDim,blockDim, stream.stream>>>(column, offset, entry_num, *out);
	CUT_CHECK_ERROR("gc_intersect");
	gc_free(offset);
}

__global__ void gc_intersect_kernel(char* pos1, char* pos2, int entry_num, char* pos)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;
	
	char4* raw1 = (char4*)pos1;
	char4* raw2 = (char4*)pos2;
	char4* raw = (char4*)pos;

	char4 v1 = raw1[ttid];
	char4 v2 = raw2[ttid];

	char4 v;
/*
	v.x = (v1.x & v2.x);
	v.y = (v1.y & v2.y);
	v.z = (v1.z & v2.z);
	v.w = (v1.w & v2.w);
*/
	if (v1.x == 1 and v2.x == 1) v.x = 1; else v.x = 0;
	if (v1.y == 1 and v2.y == 1) v.y = 1; else v.y = 0;
	if (v1.z == 1 and v2.z == 1) v.z = 1; else v.z = 0;
	if (v1.w == 1 and v2.w == 1) v.w = 1; else v.w = 0;

	raw[ttid] = v;
}

void gc_intersect(gcStream_t stream, char* pos1, char* pos2, int entry_num, char* pos)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	gc_intersect_kernel<<<gridDim, blockDim, stream.stream>>>(pos1, pos2, entry_num, pos);
	CUT_CHECK_ERROR("gc_intersect");
}
__global__ void gc_filter1_kernel(int* column, int* pos_list, int entry_num)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;
	
	int4* raw = (int4*)column;
	int4 v = raw[ttid];
	int4* raw1 = (int4*)pos_list;
	int4 v2 = raw1[ttid];
	if (v2.x == 0) v.x = 0;
	if (v2.y == 0) v.y = 0;
	if (v2.z == 0) v.z = 0;
	if (v2.w == 0) v.w = 0;
	raw[ttid] = v;
}

void gc_filter1(gcStream_t stream, int* column, int* pos_list, int entry_num)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	gc_filter1_kernel<<<gridDim, blockDim, stream.stream>>>(column, pos_list, entry_num);
	CUT_CHECK_ERROR("gc_filter1");
}

__global__ void gc_cal_q14_kernel(float *price, char* discount, char* type, int* lpartkey_offset_in, int* lpartkey_offset_out, int* lpartkey_len, char* pos_vector, int* lpartkey_pos_list, int* ppartkey_pos_list, int entry_num, int centry_num, float* out1, char* out2)
{
	int ttid = TID;
	if (ttid >= centry_num) return;

	int lpos = lpartkey_pos_list[ttid];	
	int ppos = ppartkey_pos_list[ttid];	

	int llen = lpartkey_len[lpos];
	int loffset = lpartkey_offset_in[lpos];


	for (int i = 0; i < llen; i++)
	{
		int pos = loffset + i;
		char c = pos_vector[pos / 8];
		float o1 = 0.0f;
		char o2 = 0;
		if (c & (0x80 >> (pos%8 - 1)))
		{
			float price1 = price[pos];
			float discount1 = (float)discount[pos] / 100.0;
			o1 = (1 - discount1) * price1;
			if (type[ppos] >= 125) o2 = 1;
		}
		out1[pos] = o1;
		out2[pos] = o2;
	}
}

void gc_cal_q14(gcStream_t stream, float* price, char* discount, char* type, int* lpartkey_offset_in, 
		int* lpartkey_offset_out, int* lpartkey_len, char* pos_vector, int* lpartkey_pos_list, int* ppartkey_pos_list,
		int entry_num, int centry_num, float* out1, char* out2)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(centry_num, blockDim.x), blockDim.x);
	gc_cal_q14_kernel<<<gridDim, blockDim, stream.stream>>>(price, discount, type, lpartkey_offset_in, lpartkey_offset_out, lpartkey_len, pos_vector, lpartkey_pos_list, ppartkey_pos_list, entry_num, centry_num, out1, out2);
	CUT_CHECK_ERROR("gc_pos_vector_hist");

}

__global__ void gc_cal_q14_final_kernel(float* out1, char* out2, int entry_num, float sum, float* out)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	float4* raw1 = (float4*)out1;
	char4* raw2 = (char4*)out2;

	float4 v1 = raw1[ttid];
	char4 v2 = raw2[ttid];
	
	float4 o;

	if (v2.x == 1) o.x = 100.0f * v1.x; else o.x = 0.0f;
	if (v2.y == 1) o.y = 100.0f * v1.y; else o.y = 0.0f;
	if (v2.z == 1) o.z = 100.0f * v1.z; else o.z = 0.0f;
	if (v2.w == 1) o.w = 100.0f * v1.w; else o.w = 0.0f;

	float4* out4 = (float4*)out;
	out4[ttid] = o;
}

void gc_cal_q14_final(gcStream_t stream, float* out1, char* out2, int entry_num, float sum, float* out)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	gc_cal_q14_final_kernel<<<gridDim, blockDim, stream.stream>>>(out1, out2, entry_num, sum, out);
	CUT_CHECK_ERROR("gc_pos_vector_hist");

}


/*
__global__ void gc_cal_q14_kernel(float *price, char* discount, char* type, int* lpartkey_offset_in, int* lpartkey_offset_out, int* lpartkey_len, char* pos_vector, int* lpartkey_pos_list, int* ppartkey_pos_list, int entry_num, int centry_num, float* out1, float* out2)
{
	int ttid = TID;
	if (ttid >= centry_num) return;

	int lpos = lpartkey_pos_list[ttid];	
	int ppos = ppartkey_pos_list[ttid];	

	int llen = lpartkey_len[ttid];
	int loffset = lpartkey_offset_in[lpos];

	for (int i = 0; i < llen; i++)
	{
		int pos = loffset + i;
		char c = pos_vector[pos / 8];
		float o1 = 0.0f;
		float o2 = 0.0f;
		if (c & (0x80 >> (c%8 - 1)))
		{
			float price1 = price[pos];
			float discount1 = (float)discount[pos] / 100.0;
			o2 = (1 - discount1) * price1;
			if (type[ppos] >= 125) o1 = o2;
		}
		out1[pos] = o1;
		out2[pos] = o2;
	}
}

void gc_cal_q14(gcStream_t stream, float* price, char* discount, char* type, int* lpartkey_offset_in, 
		int* lpartkey_offset_out, int* lpartkey_len, char* pos_vector, int* lpartkey_pos_list, int* ppartkey_pos_list,
		int entry_num, int centry_num, float* out1, float* out2)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(centry_num, blockDim.x), blockDim.x);
	gc_cal_q14_kernel<<<gridDim, blockDim, stream.stream>>>(price, discount, type, lpartkey_offset_in, lpartkey_offset_out, lpartkey_len, pos_vector, lpartkey_pos_list, ppartkey_pos_list, entry_num, centry_num, out1, out2);
	CUT_CHECK_ERROR("gc_pos_vector_hist");

}
*/
__global__ void gc_pos_vector_hist_kernel(int* column, char* val_vector, int centry_num, int* hist)
{
	int ttid = TID;
	if (ttid >= CEIL(centry_num, 4)) return;

	int4* raw = (int4*)column;
	int4 v = raw[ttid];
	int4* out = (int4*)hist;
	int4 o;

	o.x = val_vector[v.x];
	o.y = val_vector[v.y];
	o.z = val_vector[v.z];
	o.w = val_vector[v.w];

	out[ttid] = o;
}
__global__ void gc_pos_vector_kernel(int* column, int* hist, int* offset, int centry_num, int* pos_list)
{
	int ttid = TID;
	if (ttid >= CEIL(centry_num, 4)) return;

	int4* offset4 = (int4*)offset;
	int4* hist4 = (int4*)hist;
	int4  o = offset4[ttid];
	int4 h = hist4[ttid];

	if (h.x) pos_list[o.x] = ttid * 4;
	if (h.y) pos_list[o.y] = ttid * 4 + 1;
	if (h.z) pos_list[o.z] = ttid * 4 + 2;
	if (h.w) pos_list[o.w] = ttid * 4 + 3;
}


void gc_pos_vector(gcStream_t stream, int* column, char* val_vector, int centry_num, int** pos_list, int* num)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);

	int* hist = (int*)gc_malloc(sizeof(int) * CEIL(centry_num, 4) * 4);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(centry_num, 4), blockDim.x), blockDim.x);
	gc_pos_vector_hist_kernel<<<gridDim, blockDim, stream.stream>>>(column, val_vector, centry_num, hist);
	CUT_CHECK_ERROR("gc_pos_vector_hist");
	int* offset = (int*)gc_malloc(sizeof(int) * CEIL(centry_num, 4) * 4);
	*num = prefixSum(hist, centry_num, offset, EXCLUSIVE);
printf("%d \n", *num);
	*pos_list = (int*)gc_malloc(*num * sizeof(int));
	gc_pos_vector_kernel<<<gridDim, blockDim, stream.stream>>>(column, hist, offset, centry_num, *pos_list);
	CUT_CHECK_ERROR("gc_pos_vector");

	gc_free(offset);
	gc_free(hist);
}

__global__ void gc_val_vector_kernel(int* gpu_column, int entry_num, char* gpu_val_vector)
{
	int ttid = TID;
	if (ttid >= entry_num) return;

	int v = gpu_column[ttid];
	
	gpu_val_vector[v] = 1;
}

void gc_val_vector(gcStream_t stream, int* gpu_column, int entry_num, int max_val, char* gpu_val_vector)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	cudaMemset(gpu_val_vector, 0, max_val);

	THREAD_CONF(gridDim, blockDim, CEIL(entry_num, blockDim.x), blockDim.x);
	gc_val_vector_kernel<<<gridDim, blockDim, stream.stream>>>(gpu_column, entry_num, gpu_val_vector);
	CUT_CHECK_ERROR("gc_val_vector");
}

__global__ void gc_select_char_ng_lt(char* gpu_column, int entry_num, char low, char high, char* gpu_pos_vector)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 8)) return;

	char4* raw4 = (char4*)gpu_column;	
	int index = ttid + ttid;
	char4 v = raw4[index];
	char c = 0;

	if (v.x < high) c |= 0x80;
	if (v.y < high) c |= 0x40;
	if (v.z < high) c |= 0x20;
	if (v.w < high) c |= 0x10;

	v = raw4[index+1];
	if (v.x < high) c |= 0x08;
	if (v.y < high) c |= 0x04;
	if (v.z < high) c |= 0x02;
	if (v.w < high) c |= 0x01;

	gpu_pos_vector[ttid] = c;
}

__global__ void gc_select_char_ge_le(char* gpu_column, int entry_num, char low, char high, char* gpu_pos_vector)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 8)) return;

	char4* raw4 = (char4*)gpu_column;	
	int index = ttid + ttid;
	char4 v = raw4[index];
	char c = 0;

	if (v.x >= low && v.x <= high) c |= 0x80;
	if (v.y >= low && v.y <= high) c |= 0x40;
	if (v.z >= low && v.z <= high) c |= 0x20;
	if (v.w >= low && v.w <= high) c |= 0x10;

	v = raw4[index+1];
	if (v.x >= low && v.x <= high) c |= 0x08;
	if (v.y >= low && v.y <= high) c |= 0x04;
	if (v.z >= low && v.z <= high) c |= 0x02;
	if (v.w >= low && v.w <= high) c |= 0x01;

	gpu_pos_vector[ttid] = c;
}

__global__ void gc_select_char_gt_lt(char* gpu_column, int entry_num, char low, char high, char* gpu_pos_vector)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 8)) return;

	char4* raw4 = (char4*)gpu_column;	
	int index = ttid + ttid;
	char4 v = raw4[index];
	char c = 0;

	if (v.x > low && v.x < high) c |= 0x80;
	if (v.y > low && v.y < high) c |= 0x40;
	if (v.z > low && v.z < high) c |= 0x20;
	if (v.w > low && v.w < high) c |= 0x10;

	v = raw4[index+1];
	if (v.x > low && v.x < high) c |= 0x08;
	if (v.y > low && v.y < high) c |= 0x04;
	if (v.z > low && v.z < high) c |= 0x02;
	if (v.w > low && v.w < high) c |= 0x01;

	gpu_pos_vector[ttid] = c;
}

void gc_select_char(gcStream_t stream, char* gpu_column, int entry_num, char low, char op_low, char high, char op_high, char* gpu_pos_vector)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 8), blockDim.x), blockDim.x);

	if (op_low == GT && op_high == LT)
	{
		gc_select_char_gt_lt<<<gridDim, blockDim, stream.stream>>>(gpu_column, entry_num, low, high, gpu_pos_vector);
		CUT_CHECK_ERROR("gc_select_float_gt_lt");
	}
	else if (op_low == GE && op_high == LE)
	{
		gc_select_char_ge_le<<<gridDim, blockDim, stream.stream>>>(gpu_column, entry_num, low, high, gpu_pos_vector);
		CUT_CHECK_ERROR("gc_select_float_gt_lt");
	}
	else if (op_low == NG && op_high == LT)
	{
		gc_select_char_ng_lt<<<gridDim, blockDim, stream.stream>>>(gpu_column, entry_num, low, high, gpu_pos_vector);
		CUT_CHECK_ERROR("gc_select_float_gt_lt");
	}
}
__global__ void gc_select_short_ge_lt(short* gpu_column, int entry_num, short low, short high, char* gpu_pos_vector)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 8)) return;

	short4* raw4 = (short4*)gpu_column;	
	int index = ttid + ttid;
	short4 v = raw4[index];
	char c = 0;

	if (v.x >= low && v.x < high) c |= 0x80;
	if (v.y >= low && v.y < high) c |= 0x40;
	if (v.z >= low && v.z < high) c |= 0x20;
	if (v.w >= low && v.w < high) c |= 0x10;

	v = raw4[index+1];
	if (v.x >= low && v.x < high) c |= 0x08;
	if (v.y >= low && v.y < high) c |= 0x04;
	if (v.z >= low && v.z < high) c |= 0x02;
	if (v.w >= low && v.w < high) c |= 0x01;

	gpu_pos_vector[ttid] = c;
}
void gc_select_short(gcStream_t stream, short* gpu_column, int entry_num, short low, char op_low, short high, char op_high, char* gpu_pos_vector)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 8), blockDim.x), blockDim.x);

	if (op_low == GE && op_high == LT)
	{
		gc_select_short_ge_lt<<<gridDim, blockDim, stream.stream>>>(gpu_column, entry_num, low, high, gpu_pos_vector);
		CUT_CHECK_ERROR("gc_select_float_gt_lt");
	}
}


__global__ void gc_select_int_ng_lt(int* gpu_column, int entry_num, int low, int high, char* gpu_pos_vector)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 8)) return;

	int4* raw4 = (int4*)gpu_column;	
	int index = ttid + ttid;
	int4 v = raw4[index];
	char c = 0;

	if (v.x < high) c |= 0x80;
	if (v.y < high) c |= 0x40;
	if (v.z < high) c |= 0x20;
	if (v.w < high) c |= 0x10;

	v = raw4[index+1];
	if (v.x < high) c |= 0x08;
	if (v.y < high) c |= 0x04;
	if (v.z < high) c |= 0x02;
	if (v.w < high) c |= 0x01;

	gpu_pos_vector[ttid] = c;
}

__global__ void gc_select_int_ge_le(int* gpu_column, int entry_num, int low, int high, char* gpu_pos_vector)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 8)) return;

	int4* raw4 = (int4*)gpu_column;	
	int index = ttid + ttid;
	int4 v = raw4[index];
	char c = 0;

	if (v.x >= low && v.x <= high) c |= 0x80;
	if (v.y >= low && v.y <= high) c |= 0x40;
	if (v.z >= low && v.z <= high) c |= 0x20;
	if (v.w >= low && v.w <= high) c |= 0x10;

	v = raw4[index+1];
	if (v.x >= low && v.x <= high) c |= 0x08;
	if (v.y >= low && v.y <= high) c |= 0x04;
	if (v.z >= low && v.z <= high) c |= 0x02;
	if (v.w >= low && v.w <= high) c |= 0x01;

	gpu_pos_vector[ttid] = c;
}

__global__ void gc_select_int_gt_ng(int* gpu_column, int entry_num, int low, int high, char* gpu_pos_vector)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 8)) return;

	int4* raw4 = (int4*)gpu_column;	
	int index = ttid + ttid;
	int4 v = raw4[index];
	char c = 0;

	if (v.x > low) c |= 0x80;
	if (v.y > low) c |= 0x40;
	if (v.z > low) c |= 0x20;
	if (v.w > low) c |= 0x10;

	v = raw4[index+1];
	if (v.x > low) c |= 0x08;
	if (v.y > low) c |= 0x04;
	if (v.z > low) c |= 0x02;
	if (v.w > low) c |= 0x01;

	gpu_pos_vector[ttid] = c;
}
__global__ void gc_select_int_ge_lt(int* gpu_column, int entry_num, int low, int high, char* gpu_pos_vector)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 8)) return;

	int4* raw4 = (int4*)gpu_column;	
	int index = ttid + ttid;
	int4 v = raw4[index];
	char c = 0;

	if (v.x >= low && v.x < high) c |= 0x80;
	if (v.y >= low && v.y < high) c |= 0x40;
	if (v.z >= low && v.z < high) c |= 0x20;
	if (v.w >= low && v.w < high) c |= 0x10;

	v = raw4[index+1];
	if (v.x >= low && v.x < high) c |= 0x08;
	if (v.y >= low && v.y < high) c |= 0x04;
	if (v.z >= low && v.z < high) c |= 0x02;
	if (v.w >= low && v.w < high) c |= 0x01;

	gpu_pos_vector[ttid] = c;
}

void gc_select_int(gcStream_t stream, int* gpu_column, int entry_num, int low, char op_low, int high, char op_high, char* gpu_pos_vector)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 8), blockDim.x), blockDim.x);

	if (op_low == GE && op_high == LT)
	{
		gc_select_int_ge_lt<<<gridDim, blockDim, stream.stream>>>(gpu_column, entry_num, low, high, gpu_pos_vector);
		CUT_CHECK_ERROR("gc_select_float_gt_lt");
	}
	else if (op_low == GE && op_high == LE)
	{
		gc_select_int_ge_le<<<gridDim, blockDim, stream.stream>>>(gpu_column, entry_num, low, high, gpu_pos_vector);
		CUT_CHECK_ERROR("gc_select_float_gt_lt");
	}
	else if (op_low == NG && op_high == LT)
	{
		gc_select_int_ng_lt<<<gridDim, blockDim, stream.stream>>>(gpu_column, entry_num, low, high, gpu_pos_vector);
		CUT_CHECK_ERROR("gc_select_float_gt_lt");
	}
	else if (op_low == GT && op_high == NG)
	{
		gc_select_int_gt_ng<<<gridDim, blockDim, stream.stream>>>(gpu_column, entry_num, low, high, gpu_pos_vector);
		CUT_CHECK_ERROR("gc_select_float_gt_lt");
	}
}
__global__ void gc_select_float_gt_lt(float* gpu_column, int entry_num, float low, float high, char* gpu_pos_vector)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 8)) return;

	float4* raw4 = (float4*)gpu_column;	
	int index = ttid + ttid;
	float4 v = raw4[index];
	char c = 0;

	if (v.x > low && v.x < high) c |= 0x80;
	if (v.y > low && v.y < high) c |= 0x40;
	if (v.z > low && v.z < high) c |= 0x20;
	if (v.w > low && v.w < high) c |= 0x10;

	v = raw4[index+1];
	if (v.x > low && v.x < high) c |= 0x08;
	if (v.y > low && v.y < high) c |= 0x04;
	if (v.z > low && v.z < high) c |= 0x02;
	if (v.w > low && v.w < high) c |= 0x01;

	gpu_pos_vector[ttid] = c;
}


void gc_select_float(gcStream_t stream, float* gpu_column, int entry_num, float low, char op_low, float high, char op_high, char* gpu_pos_vector)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 8), blockDim.x), blockDim.x);

	if (op_low == GT && op_high == LT)
	{
		gc_select_float_gt_lt<<<gridDim, blockDim, stream.stream>>>(gpu_column, entry_num, low, high, gpu_pos_vector);
		CUT_CHECK_ERROR("gc_select_float_gt_lt");
	}
}

__global__ void gc_filter_kernel(char* pos_vector1, char* pos_vector2, char* pos_vector3, int entry_num, char* pos_vector)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 32)) return;

	char4* raw1 = (char4*)pos_vector1;
	char4* raw2 = (char4*)pos_vector2;
	char4* raw3 = (char4*)pos_vector3;
	char4* out = (char4*)pos_vector;

	char4 v1 = raw1[ttid];
	char4 v2 = raw2[ttid];
	char4 v3 = raw3[ttid];

	char4 o;

	o.x = (v1.x & v2.x & v3.x);
	o.y = (v1.y & v2.y & v3.y);
	o.z = (v1.z & v2.z & v3.z);
	o.w = (v1.w & v2.w & v3.w);

	out[ttid] = o;
}
 
void gc_filter(gcStream_t stream, char* pos_vector1, char* pos_vector2, char* pos_vector3, char* pos_vector, int entry_num)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 32), blockDim.x), blockDim.x);

	gc_filter_kernel<<<gridDim, blockDim, stream.stream>>>(pos_vector1, pos_vector2, pos_vector3, entry_num, pos_vector);
	CUT_CHECK_ERROR("gc_filter_kernel");
}

__global__ void gc_filter_float_product(float* column1, float* column2, int entry_num, char* pos_vector, float* product)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 8)) return;

	int index = ttid + ttid;
	char c = pos_vector[ttid];
	float4* raw1 = (float4*)column1;
	float4* raw2 = (float4*)column2;
	float4 v1 = raw1[index];
	float4 v2 = raw2[index];
	float4* out = (float4*)product;
	float4 o;

	if (c & 0x80) o.x = v1.x * v2.x; else o.x = 0.0f;
	if (c & 0x40) o.y = v1.y * v2.y; else o.y = 0.0f;
	if (c & 0x20) o.z = v1.z * v2.z; else o.z = 0.0f;
	if (c & 0x10) o.w = v1.w * v2.w; else o.w = 0.0f;

	out[index] = o;

	v1 = raw1[index+1];
	v2 = raw2[index+1];

	if (c & 0x08) o.x = v1.x * v2.x; else o.x = 0.0f;
	if (c & 0x04) o.y = v1.y * v2.y; else o.y = 0.0f;
	if (c & 0x02) o.z = v1.z * v2.z; else o.z = 0.0f;
	if (c & 0x01) o.w = v1.w * v2.w; else o.w = 0.0f;

	out[index+1] = o;
}

void gc_filter_float_value(gcStream_t stream, float* column1, float* column2, int entry_num, char* pos_vector, float* out)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 8), blockDim.x), blockDim.x);

	gc_filter_float_product<<<gridDim, blockDim, stream.stream>>>(column1, column2, entry_num, pos_vector, out);
	CUT_CHECK_ERROR("gc_filter_float_product");

/*
	int* hist = (int*)gc_malloc(sizeof(int) * CEIL(entry_num, 8) * 8);
	gc_filter_float_hist_kernel<<<gridDim, blockDim, stream.stream>>>(pos_vector, hist)
	CUT_CHECK_ERROR("gc_filter_float_hist_kernel");

	int* offset = (int*)gc_malloc( sizeof(int) * CEIL(entry_num,8) * 8);
	prefixSum();
	gc_free(offset);
	gc_free(hist);
*/
}


__global__ void gc_filter_float_product2(float* column1, char* column2, int entry_num, char* pos_vector, float* product)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 8)) return;

	int index = ttid + ttid;
	char c = pos_vector[ttid];
	float4* raw1 = (float4*)column1;
	char4* raw2 = (char4*)column2;
	float4 v1 = raw1[index];
	char4 v2 = raw2[index];
	float4* out = (float4*)product;
	float4 o;
	int4 v22;
	 v22.x = (int)v2.x;
	 v22.y = (int)v2.y;
	 v22.z = (int)v2.z;
	 v22.w = (int)v2.w;

	if (c & 0x80) o.x = v1.x * (float)v22.x; else o.x = 0.0f;
	if (c & 0x40) o.y = v1.y * (float)v22.y; else o.y = 0.0f;
	if (c & 0x20) o.z = v1.z * (float)v22.z; else o.z = 0.0f;
	if (c & 0x10) o.w = v1.w * (float)v22.w; else o.w = 0.0f;

	out[index] = o;

	v1 = raw1[index+1];
	v2 = raw2[index+1];

	if (c & 0x08) o.x = v1.x * (float)v22.x; else o.x = 0.0f;
	if (c & 0x04) o.y = v1.y * (float)v22.y; else o.y = 0.0f;
	if (c & 0x02) o.z = v1.z * (float)v22.z; else o.z = 0.0f;
	if (c & 0x01) o.w = v1.w * (float)v22.w; else o.w = 0.0f;

	out[index+1] = o;
}

void gc_filter_float_value2(gcStream_t stream, float* column1, char* column2, int entry_num, char* pos_vector, float* out)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 8), blockDim.x), blockDim.x);

	gc_filter_float_product2<<<gridDim, blockDim, stream.stream>>>(column1, column2, entry_num, pos_vector, out);
	CUT_CHECK_ERROR("gc_filter_float_product");

/*
	int* hist = (int*)gc_malloc(sizeof(int) * CEIL(entry_num, 8) * 8);
	gc_filter_float_hist_kernel<<<gridDim, blockDim, stream.stream>>>(pos_vector, hist)
	CUT_CHECK_ERROR("gc_filter_float_hist_kernel");

	int* offset = (int*)gc_malloc( sizeof(int) * CEIL(entry_num,8) * 8);
	prefixSum();
	gc_free(offset);
	gc_free(hist);
*/
}
//=============================================================================
//Utils
//=============================================================================
int prefixSum(int* input, int num, int* output, char flag)
{
    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_INT;
    config.algorithm = CUDPP_SCAN;
if (flag == EXCLUSIVE)
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
else
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    
    CUDPPHandle scanplan = 0;
    CUDPPResult result = cudppPlan(&scanplan, config, num, 1, 0);  

    if (CUDPP_SUCCESS != result)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }
	cudppScan(scanplan, (void*)output, (void*)input, num);

    result = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }

	int last_offset = 0;
	int last_hist = 0;
	cudaMemcpy(&last_offset, &output[num -1], sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&last_hist, &input[num -1], sizeof(int), cudaMemcpyDeviceToHost);
	return (last_offset + last_hist);
}
float sumFloat2(float* input, int num, float* output)
{
    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    
    CUDPPHandle scanplan = 0;
    CUDPPResult result = cudppPlan(&scanplan, config, num, 1, 0);  

    if (CUDPP_SUCCESS != result)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }
	cudppScan(scanplan, (void*)output, (void*)input, num);

    result = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }

	float last_offset = 0;
	cudaMemcpy(&last_offset, &output[num -1], sizeof(float), cudaMemcpyDeviceToHost);
	return last_offset;
}

float sumFloat(float* input, int num)
{
	float* input2 = 0;
	cudaMallocHost((void**)&input2, (sizeof(float)*num));
	cudaMemcpy(input2, input, sizeof(float)*num, cudaMemcpyDeviceToHost);	
	float sum = 0.0f;
	for (int i = 0; i < num; i++)
	{
		sum += input2[i];
		//printf("%f, %f\n", input2[i], sum);
	}
	cudaFreeHost(input2);
	return sum;
/*
    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    
    CUDPPHandle scanplan = 0;
    CUDPPResult result = cudppPlan(&scanplan, config, num, 1, 0);  

    if (CUDPP_SUCCESS != result)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }
	cudppScan(scanplan, (void*)output, (void*)input, num);

    result = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }

	float last_offset = 0;
	cudaMemcpy(&last_offset, &output[num -1], sizeof(float), cudaMemcpyDeviceToHost);
	return last_offset;
*/
}

//=============================================================================
//GPU compression
//=============================================================================
//-----------------------------------------------------------------------------
//nsv
//-----------------------------------------------------------------------------
__device__ int byteNum(int num)
{
	if (num > TWO_BYTE) return 3;
	if (num > ONE_BYTE) return 2;
	return 1;
}
__device__ int byteNumLong(long num)
{
	if (num > THREE_BYTE) return 4;
	if (num > ONE_BYTE) return 2;
	return 1;
}

__global__ void gc_compress_nsv_long_kernel1(long* gpu_ubuf, int entry_num, int* gpu_hist)
{
	int ttid = TID;
	if (ttid >= entry_num) return;

	long v = gpu_ubuf[ttid];
	gpu_hist[ttid] = byteNumLong(v);	
}

__global__ void gc_compress_nsv_long_kernel2(long* gpu_ubuf, int* gpu_offset, int entry_num, char* gpu_value, char* gpu_len)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	__shared__ long2 sbuf[256];
	long2* raw = (long2*)gpu_ubuf;
	sbuf[threadIdx.x] = raw[ttid*2];
	__syncthreads();
	int hist; 
	int4* offset4 = (int4*)gpu_offset;
	int4 offset = offset4[ttid];
	char len = 0;
	char* src = (char*)&sbuf[threadIdx.x].x;
	hist = byteNumLong(sbuf[threadIdx.x].x);
	for (int i = 0; i < hist; ++i)
		gpu_value[offset.x + i] = src[i];	
	len |= ((char)hist << 6);

	src = (char*)&sbuf[threadIdx.x].y;
	hist = byteNumLong(sbuf[threadIdx.x].y);
	for (int i = 0; i < hist; ++i)
		gpu_value[offset.y + i] = src[i];	
	len |= ((char)hist << 4);

	sbuf[threadIdx.x] = raw[ttid*2+1];
	__syncthreads();
	src = (char*)&sbuf[threadIdx.x].x;
	hist = byteNumLong(sbuf[threadIdx.x].x);
	for (int i = 0; i < hist; ++i)
		gpu_value[offset.z + i] = src[i];	
	len |= ((char)hist << 2);

	src = (char*)&sbuf[threadIdx.x].y;
	hist = byteNumLong(sbuf[threadIdx.x].y);
	for (int i = 0; i < hist; ++i)
		gpu_value[offset.w + i] = src[i];	
	len |= ((char)hist);
	gpu_len[ttid] = len;
}

void gc_compress_nsv_long(gcStream_t Stream, long* gpu_ubuf, int entry_num, char** gpu_value, char* gpu_len, int* size) 
{
	int threadNum = CEIL(entry_num, 4);

	//step 1: hist
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(entry_num, blockDim.x), blockDim.x);
	int* gpu_hist = (int*)gc_malloc(sizeof(int) * entry_num);
	gc_compress_nsv_long_kernel1<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_hist);
	CUT_CHECK_ERROR("gc_compress_nsv_kernel1");

//cudaThreadSynchronize();
	//step 2: prefix sum
	int* gpu_offset = (int*)gc_malloc(sizeof(int) * entry_num);
	int totalSize = prefixSum(gpu_hist, entry_num, gpu_offset, EXCLUSIVE);
	*size = totalSize;

//cudaThreadSynchronize();
/*
	int* cpu_offset = (int*)malloc(sizeof(int)*entry_num);
	int* cpu_hist = (int*)malloc(sizeof(int)*entry_num);
	cudaMemcpy(cpu_offset, gpu_offset, sizeof(int)*entry_num, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_hist, gpu_hist, sizeof(int)*entry_num, cudaMemcpyDeviceToHost);
	for (int i= 0; i < 10; i++)
		printf("hist:%d, offset:%d\n", cpu_hist[i], cpu_offset[i]);
	free(cpu_hist);
	free(cpu_offset);
*/
	//step 3: scatter
	THREAD_CONF(gridDim, blockDim, CEIL(threadNum, blockDim.x), blockDim.x);
	*gpu_value = (char*)gc_malloc(totalSize);
	gc_compress_nsv_long_kernel2<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, gpu_offset, entry_num, *gpu_value, gpu_len);
	CUT_CHECK_ERROR("gc_compress_nsv_kernel2");

	gc_free(gpu_hist);
	gc_free(gpu_offset);
}

__global__ void gc_compress_nsv_kernel1(int* gpu_ubuf, int entry_num, int* gpu_hist)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	int4* raw4 = (int4*)gpu_ubuf;
	int4 v = raw4[ttid];
	int o = 0;
	o += byteNum(v.x);	
	o += byteNum(v.y);	
	o += byteNum(v.z);	
	o += byteNum(v.w);	
	gpu_hist[ttid] = o;
}

__global__ void gc_compress_nsv_kernel2(int* gpu_ubuf, int* gpu_offset, int entry_num, char* gpu_value, char* gpu_len)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	int4* raw4 = (int4*)gpu_ubuf;
	__shared__ int4 sbuf[256];
	sbuf[threadIdx.x] = raw4[ttid];
	__syncthreads();

	int hist; 
	int offset = gpu_offset[ttid];

	char len = 0;

	char* src = (char*)&sbuf[threadIdx.x].x;
	hist = byteNum(sbuf[threadIdx.x].x);
	for (int i = 0; i < hist; ++i)
		gpu_value[offset + i] = src[i];	
	len |= ((char)hist << 6);
	offset += hist;

	src = (char*)&sbuf[threadIdx.x].y;
	hist = byteNum(sbuf[threadIdx.x].y);
	for (int i = 0; i < hist; ++i)
		gpu_value[offset + i] = src[i];	
	len |= ((char)hist << 4);
	offset += hist;

	src = (char*)&sbuf[threadIdx.x].z;
	hist = byteNum(sbuf[threadIdx.x].z);
	for (int i = 0; i < hist; ++i)
		gpu_value[offset + i] = src[i];	
	len |= ((char)hist << 2);
	offset += hist;

	src = (char*)&sbuf[threadIdx.x].w;
	hist = byteNum(sbuf[threadIdx.x].w);
	for (int i = 0; i < hist; ++i)
		gpu_value[offset + i] = src[i];	
	len |= ((char)hist);
	offset += hist;

	gpu_len[ttid] = len;
}

void gc_compress_nsv(gcStream_t Stream, int* gpu_ubuf, int entry_num, char** gpu_value, char* gpu_len, int* size) 
{
	int threadNum = CEIL(entry_num, 4);

	//step 1: hist
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(threadNum, blockDim.x), blockDim.x);
	int* gpu_hist = (int*)gc_malloc(sizeof(int) * threadNum);
	gc_compress_nsv_kernel1<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_hist);
	CUT_CHECK_ERROR("gc_compress_nsv_kernel1");

	//step 2: prefix sum
	int* gpu_offset = (int*)gc_malloc(sizeof(int) * threadNum);
	int totalSize = prefixSum(gpu_hist, threadNum, gpu_offset, EXCLUSIVE);
	*size = totalSize;
/*
	int* cpu_offset = (int*)malloc(sizeof(int)*entry_num);
	int* cpu_hist = (int*)malloc(sizeof(int)*entry_num);
	cudaMemcpy(cpu_offset, gpu_offset, sizeof(int)*entry_num, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_hist, gpu_hist, sizeof(int)*entry_num, cudaMemcpyDeviceToHost);
	for (int i= 0; i < 10; i++)
		printf("hist:%d, offset:%d\n", cpu_hist[i], cpu_offset[i]);
	free(cpu_hist);
	free(cpu_offset);
*/
	//step 3: scatter
	*gpu_value = (char*)gc_malloc(totalSize);
	gc_compress_nsv_kernel2<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, gpu_offset, entry_num, *gpu_value, gpu_len);
	CUT_CHECK_ERROR("gc_compress_nsv_kernel2");

	gc_free(gpu_hist);
	gc_free(gpu_offset);
}

#if 0
__global__ void gc_decompress_nsv_kernel1(char* gpu_len, int entry_num, int* gpu_hist)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	int4* hist = (int4*)gpu_hist;
	char v = gpu_len[ttid];
	int4 o; 
	o.x = ((v & 0xc0) >> 6);
	o.y = ((v & 0x30) >> 4);
	o.z = ((v & 0x0c) >> 2);
	o.w = ((v & 0x03));
	hist[ttid] = o;
}

__global__ void gc_decompress_nsv_kernel2(int* gpu_ubuf, int* gpu_offset, int entry_num, char* gpu_value, int* gpu_hist)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;
	__shared__ int4 ibuf[256];
	ibuf[threadIdx.x].x = 0;
	ibuf[threadIdx.x].y = 0;
	ibuf[threadIdx.x].z = 0;
	ibuf[threadIdx.x].w = 0;
	__syncthreads();

	int4* hist4 = (int4*)gpu_hist;
	int4 h = hist4[ttid];
	int4* offset4 = (int4*)gpu_offset;
	int4 o = offset4[ttid];

	char* cbuf = NULL;
	cbuf = (char*)&ibuf[threadIdx.x].x;
	for (int i = 0; i < h.x; i++)
		cbuf[i] = gpu_value[o.x + i];	

	cbuf = (char*)&ibuf[threadIdx.x].y;
	for (int i = 0; i < h.y; i++)
		cbuf[i] = gpu_value[o.y + i];	

	cbuf = (char*)&ibuf[threadIdx.x].z;
	for (int i = 0; i < h.z; i++)
		cbuf[i] = gpu_value[o.z + i];	

	cbuf = (char*)&ibuf[threadIdx.x].w;
	for (int i = 0; i < h.w; i++)
		cbuf[i] = gpu_value[o.w + i];	


	int4* raw4 = (int4*)gpu_ubuf;
	raw4[ttid] = ibuf[threadIdx.x];
	__syncthreads();
}

void gc_decompress_nsv(gcStream_t Stream, int* gpu_ubuf, int entry_num, char* gpu_value, char* gpu_len) 
{
	int threadNum = CEIL(entry_num, 4);

	//step 1: hist
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(threadNum, blockDim.x), blockDim.x);
	int* gpu_hist = (int*)gc_malloc(sizeof(int) * entry_num);
	gc_decompress_nsv_kernel1<<<gridDim, blockDim, Stream.stream>>>(gpu_len, entry_num, gpu_hist);
	CUT_CHECK_ERROR("gc_compress_nsv_kernel1");

	//step 2: prefix sum
	int* gpu_offset = (int*)gc_malloc(sizeof(int) * entry_num);
	int totalSize = prefixSum(gpu_hist, entry_num, gpu_offset, EXCLUSIVE);

/*
	int* cpu_offset = (int*)malloc(sizeof(int)*threadNum);
	int* cpu_hist = (int*)malloc(sizeof(int)*threadNum);
	cudaMemcpy(cpu_offset, gpu_offset, sizeof(int)*threadNum, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_hist, gpu_hist, sizeof(int)*threadNum, cudaMemcpyDeviceToHost);
	for (int i= 0; i < threadNum; i++)
		printf("hist:%d, offset:%d\n", cpu_hist[i], cpu_offset[i]);
	free(cpu_hist);
	free(cpu_offset);
*/

	//step 3: scatter
	gc_decompress_nsv_kernel2<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, gpu_offset, entry_num, gpu_value, gpu_hist);
	CUT_CHECK_ERROR("gc_compress_nsv_kernel2");

	gc_free(gpu_hist);
	gc_free(gpu_offset);
}
#endif

__global__ void gc_decompress_nsv_kernel1(char* gpu_len, int entry_num, int* gpu_hist)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	char v = gpu_len[ttid];
	int o = 0; 
	o += ((v & 0xc0) >> 6);
	o += ((v & 0x30) >> 4);
	o += ((v & 0x0c) >> 2);
	o += ((v & 0x03));
	gpu_hist[ttid] = o;
}

__global__ void gc_decompress_nsv_kernel2(int* gpu_ubuf, int* gpu_offset, int entry_num, char* gpu_value, char* gpu_len)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;
	__shared__ int4 ibuf[256];
	ibuf[threadIdx.x].x = 0;
	ibuf[threadIdx.x].y = 0;
	ibuf[threadIdx.x].z = 0;
	ibuf[threadIdx.x].w = 0;
	__syncthreads();

	char v = gpu_len[ttid];
	int h = 0; 
	int o = gpu_offset[ttid];

	char* cbuf = NULL;
	h = ((v & 0xc0) >> 6);
	cbuf = (char*)&ibuf[threadIdx.x].x;
	for (int i = 0; i < h; i++)
		cbuf[i] = gpu_value[o + i];	
	o += h;

	h = ((v & 0x30) >> 4);
	cbuf = (char*)&ibuf[threadIdx.x].y;
	for (int i = 0; i < h; i++)
		cbuf[i] = gpu_value[o + i];	
	o += h;

	h = ((v & 0x0c) >> 2);
	cbuf = (char*)&ibuf[threadIdx.x].z;
	for (int i = 0; i < h; i++)
		cbuf[i] = gpu_value[o + i];	
	o += h;

	h = ((v & 0x03));
	cbuf = (char*)&ibuf[threadIdx.x].w;
	for (int i = 0; i < h; i++)
		cbuf[i] = gpu_value[o + i];	
	o += h;


	int4* raw4 = (int4*)gpu_ubuf;
	raw4[ttid] = ibuf[threadIdx.x];
	__syncthreads();
}

void gc_decompress_nsv(gcStream_t Stream, int* gpu_ubuf, int entry_num, char* gpu_value, char* gpu_len) 
{
	int threadNum = CEIL(entry_num, 4);

	//step 1: hist
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(threadNum, blockDim.x), blockDim.x);
	int* gpu_hist = (int*)gc_malloc(sizeof(int) * threadNum);
	gc_decompress_nsv_kernel1<<<gridDim, blockDim, Stream.stream>>>(gpu_len, entry_num, gpu_hist);
	CUT_CHECK_ERROR("gc_compress_nsv_kernel1");

	//step 2: prefix sum
	int* gpu_offset = (int*)gc_malloc(sizeof(int) * threadNum);
	int totalSize = prefixSum(gpu_hist, threadNum, gpu_offset, EXCLUSIVE);
	gc_free(gpu_hist);
/*
	int* cpu_offset = (int*)malloc(sizeof(int)*threadNum);
	int* cpu_hist = (int*)malloc(sizeof(int)*threadNum);
	cudaMemcpy(cpu_offset, gpu_offset, sizeof(int)*threadNum, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_hist, gpu_hist, sizeof(int)*threadNum, cudaMemcpyDeviceToHost);
	for (int i= 0; i < threadNum; i++)
		printf("hist:%d, offset:%d\n", cpu_hist[i], cpu_offset[i]);
	free(cpu_hist);
	free(cpu_offset);
*/

	//step 3: scatter
	gc_decompress_nsv_kernel2<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, gpu_offset, entry_num, gpu_value, gpu_len);
	CUT_CHECK_ERROR("gc_compress_nsv_kernel2");

	gc_free(gpu_offset);
}
#if 0
__global__ void gc_decompress_nsv_kernel1(char* gpu_len, int entry_num, int* gpu_hist)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	char v = gpu_len[ttid];
	int4* hist4 = (int4*)gpu_hist;
	int4 o; 
	o.x = ((v & 0xc0) >> 6);
	o.y = ((v & 0x30) >> 4);
	o.z = ((v & 0x0c) >> 2);
	o.w = ((v & 0x03));
	hist4[ttid] = o;
}

__global__ void gc_decompress_nsv_kernel2(int* gpu_ubuf, int* gpu_offset, int* gpu_hist, int entry_num, char* gpu_value)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;
	__shared__ int4 ibuf[256];
	ibuf[threadIdx.x].x = 0;
	ibuf[threadIdx.x].y = 0;
	ibuf[threadIdx.x].z = 0;
	ibuf[threadIdx.x].w = 0;
	__syncthreads();

	int4* hist = (int4*)gpu_hist;
	int4 h = hist[ttid];
	int4* offset = (int4*)gpu_offset;
	int4 o = offset[ttid];

	char* cbuf = NULL;
	cbuf = (char*)&ibuf[threadIdx.x].x;
	for (int i = 0; i < h.x; i--)
		cbuf[3 - i] = gpu_value[o.x + h.x - 1 - i];	

	cbuf = (char*)&ibuf[threadIdx.x].y;
	for (int i = 0; i < h.y; i--)
		cbuf[3 - i] = gpu_value[o.y + h.y - 1 - i];	

	cbuf = (char*)&ibuf[threadIdx.x].z;
	for (int i = 0; i < h.z; i--)
		cbuf[3 - i] = gpu_value[o.z + h.z - 1 - i];	

	cbuf = (char*)&ibuf[threadIdx.x].w;
	for (int i = 0; i < h.w; i--)
		cbuf[3 - i] = gpu_value[o.w + h.w - 1 - i];	

	int4* raw4 = (int4*)gpu_ubuf;
	raw4[ttid] = ibuf[threadIdx.x];
}

void gc_decompress_nsv(gcStream_t Stream, int* gpu_ubuf, int entry_num, char* gpu_value, char* gpu_len) 
{
	int threadNum = CEIL(entry_num, 4);

	//step 1: hist
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(threadNum, blockDim.x), blockDim.x);
	int* gpu_hist = (int*)gc_malloc(sizeof(int) * entry_num);
	gc_decompress_nsv_kernel1<<<gridDim, blockDim, Stream.stream>>>(gpu_len, entry_num, gpu_hist);
	CUT_CHECK_ERROR("gc_compress_nsv_kernel1");

	//step 2: prefix sum
	int* gpu_offset = (int*)gc_malloc(sizeof(int) * entry_num);
	int totalSize = prefixSum(gpu_hist, entry_num, gpu_offset, EXCLUSIVE);

/*
	int* cpu_offset = (int*)malloc(sizeof(int)*entry_num);
	int* cpu_hist = (int*)malloc(sizeof(int)*entry_num);
	cudaMemcpy(cpu_offset, gpu_offset, sizeof(int)*entry_num, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_hist, gpu_hist, sizeof(int)*entry_num, cudaMemcpyDeviceToHost);
	for (int i= 0; i < 10; i++)
		printf("hist:%d, offset:%d\n", cpu_hist[i], cpu_offset[i]);
	free(cpu_hist);
	free(cpu_offset);
*/
	//step 3: scatter
return;
	gc_decompress_nsv_kernel2<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, gpu_offset, gpu_hist, entry_num, gpu_value);
	CUT_CHECK_ERROR("gc_compress_nsv_kernel2");

	gc_free(gpu_offset);
	gc_free(gpu_hist);
}
#endif
#if 0
__global__ void gc_compress_nsv_kernel1(int* gpu_ubuf, int entry_num, int* gpu_hist)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	int4* raw4 = (int4*)gpu_ubuf;
	int4 v = raw4[ttid];
	int4 o;
	int4* out4 = (int4*)gpu_hist;
	o.x = byteNum(v.x);	
	o.y = byteNum(v.y);	
	o.z = byteNum(v.z);	
	o.w = byteNum(v.w);	
	out4[ttid] = o;
}

__global__ void gc_compress_nsv_kernel2(int* gpu_ubuf, int* gpu_offset, int* gpu_hist, int entry_num, char* gpu_value, char* gpu_len)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	int4* raw4 = (int4*)gpu_ubuf;
	__shared__ int4 sbuf[256];
	sbuf[threadIdx.x] = raw4[ttid];
	__syncthreads();

	int4* hist4 = (int4*)gpu_hist;
	int4 hist = hist4[ttid];
	int4* offset4 = (int4*)gpu_offset;
	int4 offset = offset4[ttid];

	char len = 0;

	char* src = (char*)&sbuf[threadIdx.x].x;
	for (int i = 0; i < hist.x; ++i)
		gpu_value[offset.x + i] = src[i];	
	len |= ((char)hist.x << 6);

	src = (char*)&sbuf[threadIdx.x].y;
	for (int i = 0; i < hist.y; ++i)
		gpu_value[offset.y + i] = src[i];	
	len |= ((char)hist.x << 4);

	src = (char*)&sbuf[threadIdx.x].z;
	for (int i = 0; i < hist.z; ++i)
		gpu_value[offset.z + i] = src[i];	
	len |= ((char)hist.x << 2);

	src = (char*)&sbuf[threadIdx.x].w;
	for (int i = 0; i < hist.w; ++i)
		gpu_value[offset.w + i] = src[i];	
	len |= ((char)hist.x);

	gpu_len[ttid] = len;
}

void gc_compress_nsv(gcStream_t Stream, int* gpu_ubuf, int entry_num, char** gpu_value, char* gpu_len) 
{
	int threadNum = CEIL(entry_num, 4);

	//step 1: hist
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(threadNum, blockDim.x), blockDim.x);
	int* gpu_hist = (int*)gc_malloc(sizeof(int) * entry_num);
	gc_compress_nsv_kernel1<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_hist);
	CUT_CHECK_ERROR("gc_compress_nsv_kernel1");
return;
	//step 2: prefix sum
	int* gpu_offset = (int*)gc_malloc(sizeof(int) * entry_num);
	int totalSize = prefixSum(gpu_hist, entry_num, gpu_offset, EXCLUSIVE);

/*
	int* cpu_offset = (int*)malloc(sizeof(int)*entry_num);
	int* cpu_hist = (int*)malloc(sizeof(int)*entry_num);
	cudaMemcpy(cpu_offset, gpu_offset, sizeof(int)*entry_num, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_hist, gpu_hist, sizeof(int)*entry_num, cudaMemcpyDeviceToHost);
	for (int i= 0; i < 10; i++)
		printf("hist:%d, offset:%d\n", cpu_hist[i], cpu_offset[i]);
	free(cpu_hist);
	free(cpu_offset);
*/
	//step 3: scatter
	*gpu_value = (char*)gc_malloc(totalSize);
	gc_compress_nsv_kernel2<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, gpu_offset, gpu_hist, entry_num, *gpu_value, gpu_len);
	CUT_CHECK_ERROR("gc_compress_nsv_kernel2");

	gc_free(gpu_hist);
	gc_free(gpu_offset);
}
#endif
//-----------------------------------------------------------------------------
//bitmap
//-----------------------------------------------------------------------------
//__device__ __constant__ char gpu_mode[84];
__global__ void gc_compress_bitmap_kernel(char* gpu_ubuf, int entry_num, char* gpu_r, char* gpu_a, char* gpu_n)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 8)) return;

	int2* raw2 = (int2*)gpu_ubuf;
	__shared__ int2 sbuf[256];
	sbuf[threadIdx.x] = raw2[ttid];
	__syncthreads();
	char* raw = (char*)&sbuf[threadIdx.x];

	char b = 0;
	if (raw[0] == 'R') b |= 0x80;
	if (raw[1] == 'R') b |= 0x40;
	if (raw[2] == 'R') b |= 0x20;
	if (raw[3] == 'R') b |= 0x10;
	if (raw[4] == 'R') b |= 0x08;
	if (raw[5] == 'R') b |= 0x04;
	if (raw[6] == 'R') b |= 0x02;
	if (raw[7] == 'R') b |= 0x01;
	gpu_r[ttid] = b;
	b = 0;
	if (raw[0] == 'A') b |= 0x80;
	if (raw[1] == 'A') b |= 0x40;
	if (raw[2] == 'A') b |= 0x20;
	if (raw[3] == 'A') b |= 0x10;
	if (raw[4] == 'A') b |= 0x08;
	if (raw[5] == 'A') b |= 0x04;
	if (raw[6] == 'A') b |= 0x02;
	if (raw[7] == 'A') b |= 0x01;
	gpu_a[ttid] = b;

	b = 0;
	if (raw[0] == 'N') b |= 0x80;
	if (raw[1] == 'N') b |= 0x40;
	if (raw[2] == 'N') b |= 0x20;
	if (raw[3] == 'N') b |= 0x10;
	if (raw[4] == 'N') b |= 0x08;
	if (raw[5] == 'N') b |= 0x04;
	if (raw[6] == 'N') b |= 0x02;
	if (raw[7] == 'N') b |= 0x01;
	gpu_n[ttid] = b;
}

void gc_compress_bitmap(gcStream_t Stream, char* gpu_ubuf, int entry_num, char* gpu_r, char* gpu_a, char* gpu_n) 
{
//	char mode[84] = {0};
//	mode['R'] = 0;
//	mode['A'] = 1;
//	mode['N'] = 2;
//        cudaMemcpyToSymbol("gpu_mode", mode, 84, 0, cudaMemcpyHostToDevice);
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 8), blockDim.x), blockDim.x);
	gc_compress_bitmap_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_r, gpu_a, gpu_n);
	CUT_CHECK_ERROR("gc_compress_bitmap_kernel");
}

__global__ void gc_decompress_bitmap_kernel(char* gpu_ubuf, int entry_num, char* gpu_r, char* gpu_a, char* gpu_n)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 8)) return;

	int2* raw2 = (int2*)gpu_ubuf;
	__shared__ int2 sbuf[256];
	char* raw = (char*)&sbuf[threadIdx.x];

	char r = gpu_r[ttid];
	char a = gpu_a[ttid];
	char n = gpu_n[ttid];

	if (r & 0x80) raw[0] = 'R';
	if (r & 0x40) raw[1] = 'R';
	if (r & 0x20) raw[2] = 'R';
	if (r & 0x10) raw[3] = 'R';
	if (r & 0x08) raw[4] = 'R';
	if (r & 0x04) raw[5] = 'R';
	if (r & 0x02) raw[6] = 'R';
	if (r & 0x01) raw[7] = 'R';

	if (a & 0x80) raw[0] = 'A';
	if (a & 0x40) raw[1] = 'A';
	if (a & 0x20) raw[2] = 'A';
	if (a & 0x10) raw[3] = 'A';
	if (a & 0x08) raw[4] = 'A';
	if (a & 0x04) raw[5] = 'A';
	if (a & 0x02) raw[6] = 'A';
	if (a & 0x01) raw[7] = 'A';

	if (n & 0x80) raw[0] = 'N';
	if (n & 0x40) raw[1] = 'N';
	if (n & 0x20) raw[2] = 'N';
	if (n & 0x10) raw[3] = 'N';
	if (n & 0x08) raw[4] = 'N';
	if (n & 0x04) raw[5] = 'N';
	if (n & 0x02) raw[6] = 'N';
	if (n & 0x01) raw[7] = 'N';

	__syncthreads();

	 raw2[ttid]=sbuf[threadIdx.x];
	__syncthreads();
}

void gc_decompress_bitmap(gcStream_t Stream, char* gpu_ubuf, int entry_num, char* gpu_r, char* gpu_a, char* gpu_n) 
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 8), blockDim.x), blockDim.x);
	gc_decompress_bitmap_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_r, gpu_a, gpu_n);
	CUT_CHECK_ERROR("gc_compress_bitmap_kernel");
}
//-----------------------------------------------------------------------------
//RLE
//-----------------------------------------------------------------------------
__global__ void gc_compress_rle_kernel1(int* gpu_ubuf, int entry_num, int* hist, int* pos)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	int4* raw4 = (int4*)gpu_ubuf; 
	int4* hist4 = (int4*)hist;
	int4* pos4 = (int4*)pos;
	int4 v = raw4[ttid];
	int4 p;
	int4 h; 

	if (v.x != v.y)
	{
		h.x = 1;
		p.x = ttid * 4;	
	}
	else
	{
		h.x = 0;
		p.x = 0;
	}

	if (v.y != v.z)
	{
		h.y = 1;
		p.y = ttid * 4 + 1;	
	}
	else
	{
		h.y = 0;
		p.y = 0;
	}

	if (v.z != v.w)
	{
		h.z = 1;
		p.z = ttid * 4 + 2;	
	}
	else
	{
		h.z = 0;
		p.z = 0;
	}

	hist4[ttid] = h;
	pos4[ttid] = p;

}

__global__ void gc_compress_rle_kernel2(int* gpu_ubuf, int entry_num, int* hist, int* pos)
{
	int ttid = TID;
	if (ttid > (CEIL(entry_num, 4) - 1))  return;
	if (ttid == (CEIL(entry_num, 4) - 1)) 
	{
		hist[entry_num - 1] = 1;
		pos[entry_num - 1] = entry_num - 1;
		return;
	}

	int4* raw4 = (int4*)(gpu_ubuf + 1);
	int4* hist4 = (int4*)(hist + 1); 
	int4* pos4 = (int4*)(pos + 1); 
	int4 v = raw4[ttid];
	if (v.w != v.z)
	{
		hist4[ttid].z = 1;
		pos4[ttid].z = ttid * 4 + 3;
	}
	else
	{
		hist4[ttid].z = 0;
		pos4[ttid].z = 0;
	}
}
__global__ void gc_compress_rle_kernel3(int* gpu_ubuf, int* hist, int* offset, int entry_num, int* gpu_valbuf, int* gpu_lenbuf)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	int4* raw4 = (int4*)gpu_ubuf; 
	int4* hist4 = (int4*)hist;
	int4* offset4 = (int4*)offset;
	int4 v = raw4[ttid];
	int4 h = hist4[ttid];
	int4 o = offset4[ttid];

	if (h.x == 1)
	{
		gpu_valbuf[o.x] = v.x;		
		gpu_lenbuf[o.x] = ttid * 4;
	}
	if (h.y == 1)
	{
		gpu_valbuf[o.y] = v.y;		
		gpu_lenbuf[o.y] = ttid * 4 + 1;
	}
	if (h.z == 1)
	{
		gpu_valbuf[o.z] = v.z;		
		gpu_lenbuf[o.z] = ttid * 4 + 2;
	}
	if (h.w == 1)
	{
		gpu_valbuf[o.w] = v.w;		
		gpu_lenbuf[o.w] = ttid * 4 + 3;
	}
}
__global__ void gc_compress_rle_kernel4(int* gpu_ubuf, int entry_num, int* gpu_cbuf)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	int4* raw4 = (int4*)gpu_ubuf; 
	int4* out4 = (int4*)gpu_cbuf;
	int4 v = raw4[ttid];
	int4 o;

	o.x = v.x;
	o.y = v.y - v.x;
	o.z = v.z - v.y;
	o.w = v.w - v.z;

	out4[ttid] = o;

}

__global__ void gc_compress_rle_kernel5(int* gpu_ubuf, int entry_num, int* gpu_cbuf)
{
	int ttid = TID;
	if (ttid >= (CEIL(entry_num, 4) - 1)) return;
	if (ttid == 0) gpu_cbuf[0] = gpu_ubuf[0] + 1;

	int4* raw4 = (int4*)(gpu_ubuf + 1);
	int4* out4 = (int4*)(gpu_cbuf + 1); 
	int4 v = raw4[ttid];
	out4[ttid].w = v.w - v.z;
}


void gc_compress_rle(gcStream_t Stream, int* gpu_ubuf, int entry_num,
                     int** gpu_valbuf, int** gpu_lenbuf, int* centry_num)
{
	//step 1: get hist and pos
	int* hist = (int*)gc_malloc(sizeof(int)*entry_num);
	int* pos = (int*)gc_malloc(sizeof(int)*entry_num);
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	gc_compress_rle_kernel1<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, hist, pos);
	CUT_CHECK_ERROR("gc_compress_rle_kernel1");

	gc_compress_rle_kernel2<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, hist, pos);
	CUT_CHECK_ERROR("gc_compress_rle_kernel2");
	gc_free(pos);

	//step 2: get offset
	int* offset = (int*)gc_malloc(sizeof(int)*entry_num);
	cudaMemset(offset, 0, sizeof(int)*entry_num);
	*centry_num = prefixSum(hist, entry_num, offset, EXCLUSIVE);
	printf("centry_num:%d\n", *centry_num);

	//Step 3: get value and pos2
	int* gpu_pos2 = (int*)gc_malloc(sizeof(int) * (*centry_num));
	* gpu_valbuf = (int*)gc_malloc(sizeof(int)*(*centry_num));
	* gpu_lenbuf = (int*)gc_malloc(sizeof(int)*(*centry_num));
	gc_compress_rle_kernel3<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, hist, offset, entry_num, *gpu_valbuf, gpu_pos2);
	CUT_CHECK_ERROR("gc_compress_rle_kernel3");

	//Step 4: get len
	int cnum = *centry_num;
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(cnum, 4), blockDim.x), blockDim.x);
	gc_compress_rle_kernel4<<<gridDim, blockDim, Stream.stream>>>(gpu_pos2, *centry_num, *gpu_lenbuf);
	CUT_CHECK_ERROR("gc_compress_rle_kernel4");
	gc_compress_rle_kernel5<<<gridDim, blockDim, Stream.stream>>>(gpu_pos2, *centry_num, *gpu_lenbuf);
	CUT_CHECK_ERROR("gc_compress_rle_kernel5");

	gc_free(hist);
	gc_free(gpu_pos2);
	gc_free(offset);
}

__global__ void gc_decompress_rle_kernel(int* gpu_val, int* gpu_len, int centry_num, int* offset, int* gpu_ubuf)
{
	int ttid = TID;
	if (ttid >= centry_num) return;

	int val = gpu_val[ttid];
	int len = gpu_len[ttid];
	int off = offset[ttid];

	int* base = gpu_ubuf + off;
	for (int i = 0; i < len; i++)
		base[i] = val;
}
__global__ void gc_decompress_rle_kernel1(int* offset, int* hist, int centry_num)
{
	int ttid = TID;
	if (ttid >= centry_num) return;
	
	hist[offset[ttid]] = 1;
}
__global__ void gc_decompress_rle_kernel2(int* gpu_ubuf, int* gpu_val, int* offset, int entry_num)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;
	
	int4* offset4 = (int4*)offset;	
	int4 o4 = offset4[ttid];
	int4 o;
	o.x = gpu_val[o4.x];
	o.y = gpu_val[o4.y];
	o.z = gpu_val[o4.z];
	o.w = gpu_val[o4.w];
	int4* ubuf4 = (int4*)gpu_ubuf;
	ubuf4[ttid] = o;
}


void gc_decompress_rle(gcStream_t Stream, int** gpu_ubuf, int* entry_num, 
                     int* gpu_valbuf, int* gpu_lenbuf, int centry_num)
{
	//step 1: get offset for centry
	int* offset = (int*)gc_malloc(sizeof(int)*centry_num);
	*entry_num = prefixSum(gpu_lenbuf, centry_num, offset, EXCLUSIVE);
	printf("%d\n", *entry_num);

	//step 2: mark boundary
	int* hist = (int*)gc_malloc(sizeof(int) * (*entry_num));
	cudaMemset(hist, 0, sizeof(int) * (*entry_num));
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(centry_num, blockDim.x), blockDim.x);
	gc_decompress_rle_kernel1<<<gridDim, blockDim, Stream.stream>>>(offset, hist, centry_num);
	CUT_CHECK_ERROR("gc_compress_rle_kernel");
	gc_free(offset);
	offset = (int*)gc_malloc(sizeof(int) * (*entry_num));	
	prefixSum(hist, *entry_num, offset, INCLUSIVE);
	cudaFree(hist);

	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(*entry_num,4), blockDim.x), blockDim.x);
	*gpu_ubuf = (int*)gc_malloc(*entry_num * sizeof(int));
	gc_decompress_rle_kernel2<<<gridDim, blockDim, Stream.stream>>>(*gpu_ubuf, gpu_valbuf, offset, *entry_num);
	cudaFree(offset);
}

/*
void gc_decompress_rle(gcStream_t Stream, int** gpu_ubuf, int* entry_num, 
                     int* gpu_valbuf, int* gpu_lenbuf, int centry_num)
{
	//step 1: get offset for centry
	int* offset = (int*)gc_malloc(sizeof(int)*centry_num);
	*entry_num = prefixSum(gpu_lenbuf, centry_num, offset, EXCLUSIVE);
	*gpu_ubuf = (int*)gc_malloc(sizeof(int)* (*entry_num));

	//step 2: scatter 
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(centry_num, blockDim.x), blockDim.x);
	gc_decompress_rle_kernel<<<gridDim, blockDim>>>(gpu_valbuf, gpu_lenbuf, centry_num, offset, *gpu_ubuf);
	CUT_CHECK_ERROR("gc_compress_rle_kernel");
	
	gc_free(offset);
}
*/ 
//-----------------------------------------------------------------------------
//Scale
//-----------------------------------------------------------------------------
__global__ void gc_decompress_scale_kernel(float* gpu_ubuf, int entry_num, int* gpu_cbuf)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	float4* raw4 = (float4*)gpu_ubuf; 
	int4* out = (int4*)gpu_cbuf;
	int4 v = out[ttid];
 
	float4 o;
	o.x = (float)(v.x) / 100.0f;
	o.y = (float)(v.y) / 100.0f;
	o.z = (float)(v.z) / 100.0f;
	o.w = (float)(v.w) / 100.0f;
	raw4[ttid] = o;
}
 
void gc_decompress_scale(gcStream_t Stream, float* gpu_ubuf, int entry_num, int* gpu_cbuf) 
{
	//printf("max_num:%d, byteNum:%d\n", max_num, byteNum);
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	gc_decompress_scale_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_cbuf);
	CUT_CHECK_ERROR("gc_compress_ns_kernel");
}
__global__ void gc_compress_scale_kernel(float* gpu_ubuf, int entry_num, int* gpu_cbuf)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	float4* raw4 = (float4*)gpu_ubuf; 
	int4* out = (int4*)gpu_cbuf;
	float4 v = raw4[ttid];
 
	int4 o;
	o.x = (int)(v.x * 100.0f);
	o.y = (int)(v.y * 100.0f);
	o.z = (int)(v.z * 100.0f);
	o.w = (int)(v.w * 100.0f);
	out[ttid] = o;
}
 
void gc_compress_scale(gcStream_t Stream, float* gpu_ubuf, int entry_num, int* gpu_cbuf) 
{
	//printf("max_num:%d, byteNum:%d\n", max_num, byteNum);
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	gc_compress_scale_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_cbuf);
	CUT_CHECK_ERROR("gc_compress_ns_kernel");
}
 
//-----------------------------------------------------------------------------
//NS
//-----------------------------------------------------------------------------
__global__ void gc_compress_ns_kernel(int* gpu_ubuf, int entry_num, short* gpu_cbuf)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	int4* raw4 = (int4*)gpu_ubuf; 
	short4* out = (short4*)gpu_cbuf;
	int4 v = raw4[ttid];
 
	short4 o;
	o.x = (short)v.x;
	o.y = (short)v.y;
	o.z = (short)v.z;
	o.w = (short)v.w;
	out[ttid] = o;
}

 void gc_compress_ns2(gcStream_t Stream, int* gpu_ubuf, int entry_num, short* gpu_cbuf, int byteNum) 
{
	//printf("max_num:%d, byteNum:%d\n", max_num, byteNum);
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	gc_compress_ns_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_cbuf);
	CUT_CHECK_ERROR("gc_compress_ns_kernel");
}
  
__global__ void gc_decompress_ns2_kernel(int* gpu_ubuf, int entry_num, short* gpu_cbuf)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	int4* raw4 = (int4*)gpu_ubuf; 
	short4* out = (short4*)gpu_cbuf;
	short4 v = out[ttid];

	int4 o;
	o.x = (int)v.x;
	o.y = (int)v.y;
	o.z = (int)v.z;
	o.w = (int)v.w;
	raw4[ttid] = o;
}


void gc_decompress_ns2(gcStream_t Stream, int* gpu_ubuf, int entry_num, short* gpu_cbuf, int byteNum)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	gc_decompress_ns2_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_cbuf);
	CUT_CHECK_ERROR("gc_decompress_ns_kernel");
}

__global__ void gc_compress_ns_long_kernel(long* gpu_ubuf, int entry_num, int* gpu_cbuf)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 2)) return;

	long2* raw4 = (long2*)gpu_ubuf; 
	int2* out = (int2*)gpu_cbuf;
	long2 v = raw4[ttid];
 
	int2 o;
	o.x = (int)v.x;
	o.y = (int)v.y;
	out[ttid] = o;
}

void gc_compress_ns_long(gcStream_t Stream, long* gpu_ubuf, int entry_num, int* gpu_cbuf, int byteNum) 
{
	//printf("max_num:%d, byteNum:%d\n", max_num, byteNum);
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 2), blockDim.x), blockDim.x);
	gc_compress_ns_long_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_cbuf);
	CUT_CHECK_ERROR("gc_compress_ns_kernel");
}

__global__ void gc_compress_ns_kernel(int* gpu_ubuf, int entry_num, char* gpu_cbuf, int byteNum)
{
	int ttid = TID;
#ifdef NS3
	if (ttid >= entry_num) return;
#else
	if (ttid >= CEIL(entry_num, 4)) return;
#endif

	int4* raw4 = (int4*)gpu_ubuf; 
#ifndef NS3
	char4* out = (char4*)gpu_cbuf;
	char4 o;
	int4 v = raw4[ttid];

	o.x = (char)v.x;
	o.y = (char)v.y;
	o.z = (char)v.z;
	o.w = (char)v.w;

	out[ttid] = o;
#else
	char* src = (char*)&gpu_ubuf[ttid];
	char* dest = (char*)&gpu_cbuf[ttid*byteNum];
	for (int i = 0; i < byteNum; i++)
		dest[i] = src[i];	
#endif
}


void gc_compress_ns(gcStream_t Stream, int* gpu_ubuf, int entry_num, char* gpu_cbuf, int byteNum) 
{
	//printf("max_num:%d, byteNum:%d\n", max_num, byteNum);
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
#ifdef NS3
	THREAD_CONF(gridDim, blockDim, CEIL(entry_num , blockDim.x), blockDim.x);
#else
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num , 4), blockDim.x), blockDim.x);
#endif
	gc_compress_ns_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_cbuf, byteNum);
	CUT_CHECK_ERROR("gc_compress_ns_kernel");
}
 __global__ void gc_decompress_ns_long_kernel(long* gpu_ubuf, int entry_num, int* gpu_cbuf)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 2)) return;

	long2* raw4 = (long2*)gpu_ubuf; 
	int2* out = (int2*)gpu_cbuf;
	int2 v = out[ttid];

	long2 o;
	o.x = (long)v.x;
	o.y = (long)v.y;
	raw4[ttid] = o;
}


void gc_decompress_ns_long(gcStream_t Stream, long* gpu_ubuf, int entry_num, int* gpu_cbuf, int byteNum)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 2), blockDim.x), blockDim.x);
	gc_decompress_ns_long_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_cbuf);
	CUT_CHECK_ERROR("gc_decompress_ns_kernel");
}
__global__ void gc_decompress_ns_kernel(int* gpu_ubuf, int entry_num, char* gpu_cbuf, int byteNum)
{
	int ttid = TID;
#ifdef NS3
	if (ttid >= entry_num) return;
#else
	if (ttid >= CEIL(entry_num,4)) return;
#endif

#ifdef NS3
	char* dest = (char*)&gpu_ubuf[ttid];
	char* src = (char*)&gpu_cbuf[ttid*byteNum];
	
	//int4* raw4 = (int4*)gpu_ubuf; 
	for (int i = 0; i < byteNum; i++)
		dest[i] = src[i];	
#else
	char4* out = (char4*)gpu_cbuf;
	int4* raw4 = (int4*)gpu_ubuf;
	char4 v = out[ttid];
	v.x = v.y = v.z = v.w = 0;

	int4 o;
	o.x = o.y = o.z = o.w = 0;
/*
	o.x = (int)v.x;
	o.y = (int)v.y;
	o.z = (int)v.z;
	o.w = (int)v.w;
*/
	out[ttid] = v;
	raw4[ttid] = o;
#endif
/*
	char3* src = (char3*)gpu_cbuf;
	__shared__ char s[7200];	
	char3* sbuf = (char3*)src;
	sbuf[threadIdx.x*4] = src[ttid*4];
	sbuf[threadIdx.x*4+1] = src[ttid*4+1];
	sbuf[threadIdx.x*4+2] = src[ttid*4+2];
	sbuf[threadIdx.x*4+3] = src[ttid*4+3];
	char4* dbuf = (char4*)(s + 256 * 3);
	__syncthreads();

	int4* ddbuf = (int4*)dbuf;
	int4* out = (int4*)gpu_ubuf;

	char3*  dest = (char3*)&ddbuf[threadIdx.x].x;
	dest[0] = sbuf[threadIdx.x*4];
	dest = (char3*)&ddbuf[threadIdx.x].y;
	dest[0] = sbuf[threadIdx.x*4+1];
	dest = (char3*)&ddbuf[threadIdx.x].z;
	dest[0] = sbuf[threadIdx.x*4+2];
	dest = (char3*)&ddbuf[threadIdx.x].w;
	dest[0] = sbuf[threadIdx.x*4+3];
	__syncthreads();

	out[ttid] = ddbuf[threadIdx.x];
*/
}


void gc_decompress_ns(gcStream_t Stream, int* gpu_ubuf, int entry_num, char* gpu_cbuf, int byteNum)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
#ifdef NS3
	THREAD_CONF(gridDim, blockDim, CEIL(entry_num, blockDim.x), blockDim.x);
#else
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
#endif
	gc_decompress_ns_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_cbuf, byteNum);
	CUT_CHECK_ERROR("gc_decompress_ns_kernel");
}
 
//-----------------------------------------------------------------------------
//dict
//-----------------------------------------------------------------------------
__device__ __constant__ char gpu_dict[56];
__global__ void gc_decompress_dict_kernel(char* gpu_ubuf, int entry_num, int* gpu_cbuf)
{
	int ttid = TID;
	if (ttid >= entry_num) return;

	int* raw = (int*)gpu_cbuf;
	int v = raw[ttid];
	long* dict4 = (long*)gpu_dict;
	long* out4 = (long*)gpu_ubuf;
	long* str;

	str = dict4 + v;
	out4[ttid] = str[0];
/*
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	int4* raw4 = (int4*)gpu_cbuf;
	int4 v = raw4[ttid];
	long* dict4 = (long*)gpu_dict;
	long* out4 = (long*)gpu_ubuf + (ttid << 2); 
	out4[0] = dict4[v.x];	
	out4[1] = dict4[v.y];	
	out4[2] = dict4[v.z];	
	out4[3] = dict4[v.w];	
*/
/*
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	int4* raw4 = (int4*)gpu_cbuf;
	int4 v = raw4[ttid];
	long* dict4 = (long*)gpu_dict;
	long* out4 = (long*)gpu_ubuf + (BLOCK_ID << 10); 
	__shared__ long sbuf[1024];
	sbuf[threadIdx.x] = dict4[v.x];	
	sbuf[threadIdx.x+1] = dict4[v.y];	
	sbuf[threadIdx.x+2] = dict4[v.z];	
	sbuf[threadIdx.x+3] = dict4[v.w];	
	__syncthreads();
	//for (int i = threadIdx.x; i < 1024; i+=256)
	for (int i = 0; i < 4; i++)
		out4[threadIdx.x + i] = sbuf[threadIdx.x+i];
	__syncthreads();
*/
}

void gc_decompress_dict(gcStream_t Stream, char* gpu_ubuf, int entry_num, int* gpu_cbuf, char* dict)
{
        cudaMemcpyToSymbol("gpu_dict", dict, 56, 0, cudaMemcpyHostToDevice);
	//printf("%d\n", sizeof(long));
//	for (int i = 0; i < 7; i++)
//		printf("%s\n", &dict[i * 8]);
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	//THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	THREAD_CONF(gridDim, blockDim, CEIL(entry_num, blockDim.x), blockDim.x);
	gc_decompress_dict_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_cbuf);
	//int ssize = sizeof(long) * 4 * 256;
	//gc_decompress_dict_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_cbuf);
	CUT_CHECK_ERROR("gc_compress_dict_kernel");
}
//-----------------------------------------------------------------------------
//SEP
//-----------------------------------------------------------------------------
__global__ void gc_decompress_sep_kernel(float* gpu_ubuf, int entry_num, int* gpu_left, int* gpu_right)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	float4* raw4 = (float4*)gpu_ubuf; 
	int4* left4 = (int4*)gpu_left;
	int4* right4 = (int4*)gpu_right;
	int4 l = left4[ttid];
	int4 r = right4[ttid];
	float4 v;

	v.x = (float)l.x + (float)r.x / 100.0f;
	v.y = (float)l.y + (float)r.y / 100.0f;
	v.z = (float)l.z + (float)r.z / 100.0f;
	v.w = (float)l.w + (float)r.w / 100.0f;

	raw4[ttid] = v;
}

void gc_decompress_sep(gcStream_t Stream, float* gpu_ubuf, int entry_num, int* gpu_left, int* gpu_right)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	gc_decompress_sep_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_left, gpu_right);
	CUT_CHECK_ERROR("gc_compress_sep_kernel");
}


__global__ void gc_compress_sep_kernel(float* gpu_ubuf, int entry_num, int* gpu_left, int* gpu_right)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	float4* raw4 = (float4*)gpu_ubuf; 
	float4 v = raw4[ttid];
	int4 l;
	int4 r; 
	
	int4* left4 = (int4*)gpu_left;
	int4* right4 = (int4*)gpu_right;

	l.x = (int)v.x;
	r.x = (int)((v.x - (float)l.x) * 100.0f);

	l.y = (int)v.y;
	r.y = (int)((v.y - (float)l.y) * 100.0f);

	l.z = (int)v.z;
	r.z = (int)((v.z - (float)l.z) * 100.0f);

	l.w = (int)v.w;
	r.w = (int)((v.w - (float)l.w) * 100.0f);

	left4[ttid] = l;
	right4[ttid] = r;
}

void gc_compress_sep(gcStream_t Stream, float* gpu_ubuf, int entry_num, int* gpu_left, int* gpu_right)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	gc_compress_sep_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_left, gpu_right);
	CUT_CHECK_ERROR("gc_compress_sep_kernel");
}

//-----------------------------------------------------------------------------
//DELTA
//-----------------------------------------------------------------------------
__global__ void gc_compress_delta_kernel1(int* gpu_ubuf, int entry_num, int* gpu_cbuf)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	int4* raw4 = (int4*)gpu_ubuf; 
	int4* out4 = (int4*)gpu_cbuf;
	int4 v = raw4[ttid];
	int4 o;

	o.x = v.x;
	o.y = v.y - v.x;
	o.z = v.z - v.y;
	o.w = v.w - v.z;

	out4[ttid] = o;

}

__global__ void gc_compress_delta_kernel2(int* gpu_ubuf, int entry_num, int* gpu_cbuf)
{
	int ttid = TID;
	if (ttid >= (CEIL(entry_num, 4))) return;

/*
	int2* raw4 = (int2*)(gpu_ubuf + 3);
	int* out4 = (gpu_cbuf + 4); 
	int2 v = raw4[ttid];
	out4[ttid] = v.y - v.x;
*/
	int4* raw4 = (int4*)(gpu_ubuf + 1);
	int4* out4 = (int4*)(gpu_cbuf + 1); 
	int4 v = raw4[ttid];
	out4[ttid].w = v.w - v.z;
}

void gc_compress_delta(gcStream_t Stream, int* gpu_ubuf, int entry_num, int* gpu_cbuf, int* first_elem)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	//printf("grid.x:%d, grid.y:%d, block.x:%d, threadNum:%d\n", gridDim.x, gridDim.y, blockDim.x,  blockDim.x * gridDim.x * gridDim.y);
	gc_compress_delta_kernel1<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_cbuf);
	CUT_CHECK_ERROR("gc_compress_delta_kernel1");
	gc_compress_delta_kernel2<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_cbuf);
	CUT_CHECK_ERROR("gc_compress_delta_kernel2");
}

void gc_decompress_delta(gcStream_t stream, int* gpu_ubuf, int entry_num, int* gpu_cbuf, int first_elem)
{
	int* first = (int*)malloc(sizeof(int));
	cudaMemcpy(first, gpu_cbuf, sizeof(int), cudaMemcpyDeviceToHost);
	*first += first_elem;
	//printf("cur_first:%d, first:%d\n", *first, first_elem);
	cudaMemcpy(gpu_cbuf, first, sizeof(int), cudaMemcpyHostToDevice);
	free(first);
prefixSum(gpu_cbuf, entry_num, gpu_ubuf, INCLUSIVE);
/*
    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_INT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    
    CUDPPHandle scanplan = 0;
    CUDPPResult result = cudppPlan(&scanplan, config, entry_num, 1, 0);  

    if (CUDPP_SUCCESS != result)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }
cudppScan(scanplan, (void*)gpu_ubuf, (void*)gpu_cbuf, entry_num);

    result = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }
*/
}


//-----------------------------------------------------------------------------
//FOR
//-----------------------------------------------------------------------------
__global__ void gc_compress_for_kernel(int* gpu_ubuf, int entry_num, int* gpu_cbuf, int reference)
{
	int ttid = TID;
	if (ttid >= CEIL(entry_num, 4)) return;

	int lastid = entry_num/4;
	if (ttid != lastid)
	{
		int4* gpu_ubuf4 = (int4*)gpu_ubuf;
		int4* gpu_cbuf4 = (int4*)gpu_cbuf;
		int4 v = gpu_ubuf4[ttid];
		v.x -= reference;
		v.y -= reference;
		v.z -= reference;
		v.w -= reference;
		gpu_cbuf4[ttid] = v;	
	}
	else
	{
	//	printf("**wenbin: lastid - %d - (%x, %x), %d\n", lastid, gridDim.x, gridDim.y, blockDim.x);
		int leftNum = entry_num % 4;
		int* gpu_ubuf_left = gpu_ubuf + ttid * 4;
		int* gpu_cbuf_left = gpu_cbuf + ttid * 4;
		for (int i = 0; i < leftNum; ++i)
			gpu_cbuf_left[i] = gpu_ubuf_left[i] - reference;	
	}
}

void gc_compress_for(gcStream_t Stream, int* gpu_ubuf, int entry_num, int* gpu_cbuf, int reference)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	gc_compress_for_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_ubuf, entry_num, gpu_cbuf, reference);
	CUT_CHECK_ERROR("gc_compress_for_kernel");
}

void gc_decompress_for(gcStream_t Stream, int* gpu_ubuf, int entry_num, int* gpu_cbuf, int reference)
{
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(CEIL(entry_num, 4), blockDim.x), blockDim.x);
	int ref = -reference;
	gc_compress_for_kernel<<<gridDim, blockDim, Stream.stream>>>(gpu_cbuf, entry_num, gpu_ubuf, ref);
	CUT_CHECK_ERROR("gc_compress_for_kernel");
}

//=============================================================================
//Stream Management
//=============================================================================
void gc_stream_start(gcStream_t* Stream)
{
	CUDA_SAFE_CALL(cudaStreamCreate((cudaStream_t*)&Stream->stream));
	CUDA_SAFE_CALL(cudaEventCreate((cudaEvent_t*)&Stream->event));
	CUDA_SAFE_CALL(cudaEventCreate((cudaEvent_t*)&Stream->start));
	CUDA_SAFE_CALL(cudaEventRecord((cudaEvent_t)Stream->start, (cudaStream_t)Stream->stream));
}

void gc_stream_stop(gcStream_t* Stream)
{
	CUDA_SAFE_CALL(cudaEventRecord((cudaEvent_t)Stream->event, (cudaStream_t)Stream->stream));
	CUDA_SAFE_CALL(cudaEventSynchronize((cudaEvent_t)Stream->event));

	float etime = 0.0f;
	cudaEventElapsedTime(&etime, Stream->start, Stream->event);
	printf("***%f ms\n", etime);
	CUDA_SAFE_CALL(cudaEventDestroy((cudaEvent_t)Stream->event));
	CUDA_SAFE_CALL(cudaEventDestroy((cudaEvent_t)Stream->start));
	CUDA_SAFE_CALL(cudaStreamDestroy((cudaStream_t)Stream->stream));
}

//=============================================================================
//Memory Management
//=============================================================================
void* gc_malloc(size_t bufsize)
{
	void* gpu_buf = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_buf, bufsize));
	return gpu_buf;
}

void gc_free(void* gpu_buf)
{
	CUDA_SAFE_CALL(cudaFree(gpu_buf));
}

void* gc_host2device(gcStream_t Stream, void* cpu_buf, size_t bufsize)
{
	void* gpu_buf = NULL;

	int round_bufsize = CEIL(bufsize, 16) * 16 + 4;

	CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_buf, round_bufsize));
	CUDA_SAFE_CALL(cudaMemcpyAsync(gpu_buf, cpu_buf, bufsize, cudaMemcpyHostToDevice, Stream.stream));

	return gpu_buf;
}

void* gc_device2host(gcStream_t Stream, void* gpu_buf, size_t bufsize)
{
	void* pinned = NULL;
	CUDA_SAFE_CALL(cudaMallocHost((void**)&pinned, bufsize));
	CUDA_SAFE_CALL(cudaMemcpyAsync(pinned, gpu_buf, bufsize, cudaMemcpyDeviceToHost, Stream.stream));
	//void* cpu_buf = malloc(bufsize);
	//memcpy(cpu_buf, pinned, bufsize);
	//CUDA_SAFE_CALL(cudaFreeHost(pinned));

	//return cpu_buf;
	return pinned;
}

//=============================================================================
//Testing
//=============================================================================


__global__ void test_kernel(int* d_input, int num)
{
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        if (tid >= num) return;

        d_input[tid] = d_input[tid] * 2;
}

extern "C"
void test_gpu(int num, int print_num)
{
        if (num < print_num) return;

        int *d_input = NULL;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_input, num * sizeof(int)));
        
        int *h_input = (int*)malloc(num * sizeof(int));
        for (int i = 0; i < num; i++)
                h_input[i] = i;

        CUDA_SAFE_CALL(cudaMemcpy(d_input, h_input, sizeof(int)*num, cudaMemcpyHostToDevice));

        int block_dim = 256;
        int grid_dim = (num / 256 + (int)(num % 256 != 0));
        test_kernel<<<grid_dim, block_dim>>>(d_input, num);

        CUDA_SAFE_CALL(cudaMemcpy(h_input, d_input, sizeof(int)*num, cudaMemcpyDeviceToHost));
        for (int i = 0; i < print_num; i++)
                printf("%d - %d\n", i, h_input[i]);

        CUDA_SAFE_CALL(cudaFree(d_input));
        free(h_input);
}

void test_malloc(int size, int nloop)
{
	unsigned timer;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
	for (int i = 0; i < nloop; i++)
	{
		char* buf;
		cudaMalloc((void**)&buf, size);
		cudaFree(buf);
	}		
	cutStopTimer(timer);
	double ctime = cutGetTimerValue(timer);
	printf("allocate&free %d bytes buf: %f ms\n", size, ctime / (double)nloop);
}

__global__ void int2char(int* intSrc, char* charSrc, int num)
{
	int ttid = TID;
	if (ttid >= num) return;
	
	__shared__ int s[256];
	//__shared__ char s[256];
//	intSrc[ttid] = (int)charSrc[ttid];
//	charSrc[ttid] = (char)intSrc[ttid];
	{
		int d;
		s[threadIdx.x] = d;
		//charSrc[ttid] = s[threadIdx.x];
		//intSrc[ttid] = charSrc[ttid];
		//charSrc[ttid] = intSrc[ttid];

		//s[threadIdx.x] = intSrc[ttid];
		//intSrc[ttid] = s[threadIdx.x];
//		intSrc[ttid] = v;

		//char v = charSrc[ttid];
		//charSrc[ttid] = ttid;

		//intSrc[ttid] = i;
//		charSrc[ttid] = i;
	}
}

#define NUM 100000000
void test_int2char()
{
	int* intSrc = (int*)gc_malloc(NUM * sizeof(int));

	char* charSrc = (char*)gc_malloc(NUM * sizeof(char));

	unsigned timer;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, CEIL(NUM, blockDim.x), blockDim.x);
	int2char<<<gridDim, blockDim>>>(intSrc, charSrc, NUM);
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("gc_intersect");
	cutStopTimer(timer);
	double atime = cutGetTimerValue(timer);
	printf("%f ms\n", atime);

	gc_free(intSrc);
	gc_free(charSrc);
}
