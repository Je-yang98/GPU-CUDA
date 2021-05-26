
EXECUTABLE := cudaSaxpy

CU_FILES   := saxpy.cu

CU_DEPS    :=

CC_FILES   := main.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++
CXXFLAGS=-O3 -Wall
ifeq ($(ARCH), Darwin)
# Building on mac
LDFLAGS=-L/usr/local/depot/cuda-10.2/lib/ -lcudart
else
# Building on Linux
LDFLAGS=-L/usr/local/cuda-10.2/lib64/ -lcudart
endif
NVCC=nvcc
NVCCFLAGS=-O3


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/saxpy.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
