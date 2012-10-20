SRC	:= gcompress_cuda.cu 
NVCC	:= nvcc
#NVCC_FLAG:= -cuda -shared --host-compilation c -g  --device-emulation
NVCC_FLAG:= -cuda -shared --host-compilation c  --ptxas-options=-v -O0
#NVCC_FLAG:= -ptx 
CFLAG	:= -c -O0
#CUDADIR	:= /home/wenbin/software/cuda
CUDADIR	:= /usr/local/cuda
NVCC_INC:= /home/wenbin/CUDA_SDK/C/common/inc
NVCC_LIB:= 
#LIB	:= -L$(CUDADIR)/lib64 -L/home/wenbin/CUDA_SDK/C/lib/ -L./cudpp/cudpp/
LIB	:= -L$(CUDADIR)/lib64 -L/home/wenbin/CUDA_SDK/C/lib/ -L/home/wenbin/2009_11_MonetDB_GPU/MonetDB-Aug2009-SuperBall-SP2/MonetDB5-server/MonetDB5/src/modules/mal/gcompress_cuda/cudpp/cudpp/ 
INC	:= -I/home/wenbin/CUDA_SDK/C/common/inc -I../
#CUDPP	:= /home/wenbin/CUDA_SDK/C/common/lib/linux/libcudpp64D_emu.a
#LINK	:= -lcutil -lcuda -lcudart 
LINK	:= -lcutil -lcudart -lcudpp

libgcompress_cuda.la: gcompress_cuda.cu.c
	libtool --mode=compile gcc -o libgcompress_cuda.lo $(CFLAG) gcompress_cuda.cu.c $(CUDPP) $(LIB) $(LINK) $(INC)
	libtool --mode=link gcc -o libgcompress_cuda.la $(CFLAG) libgcompress_cuda.lo $(LIB) $(LINK) $(INC)
gcompress_cuda.cu.c: gcompress_cuda.cu
	$(NVCC) $(NVCC_FLAG)  $(SRC) -I$(NVCC_INC)

clean:
	rm -rf *.cpp *.o *.lo .libs *.c *.la
