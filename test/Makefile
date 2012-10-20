SRC	:=  main.c
OBJ	:= ../libgcompress_cuda.la
TARGET	:= c
#CUDADIR	:= /home/wenbin/software/cuda
CUDADIR	:= /usr/local/cuda
#LIB	:= -L/usr/local/cuda/lib64 -L/home/wenbin/CUDA_SDK/C/lib/ -L../
LIB	:= -L$(CUDADIR)/lib64 -L/home/wenbin/CUDA_SDK/C/lib/ -L../ -L../cudpp/cudpp/
#INC	:= -I/usr/local/cuda/include -I/home/wenbin/CUDA_SDK/C/common/inc -I../ 
INC	:= -I$(CUDADIR)/include -I/home/wenbin/CUDA_SDK/C/common/inc -I../ 
CUDPP	:=/home/wenbin/CUDA_SDK/C/common/lib/linux
#LINK	:= -lcutil -lcuda -lcudart 
LINK	:= -lcutil -lcudart  -lcudpp
DEP	:= test.h
CFLAG	:= -std=c99 -g
$(TARGET): $(SRC)  $(DEP) $(OBJ)
	libtool --mode=link gcc $(CFLAG) $(OBJ) $(INC) -o $(TARGET) $(SRC) $(LINK) $(LIB)
