BLIS_INSTALL_DIR := $(HOME)/blis

INC   := src/

BLIS_LIB  := $(BLIS_INSTALL_DIR)/lib/libblis.a
BLIS_INC  := $(BLIS_INSTALL_DIR)/include/blis

CC         := gcc
LINKER     := $(CC)
CFLAGS     := -O3 -I$(BLIS_INC) -I$(INC) -m64 -mavx2 -mfma \
              -mfpmath=sse -std=c99 -march=core-avx2 \
              -D_POSIX_C_SOURCE=200112L
CDEBUG     := -g
LDFLAGS    := -lm -lpthread

SRC = src/test.c src/gemm.c src/gemm_fp32.c src/util.c
OBJS = $(SRC:.c=.o)

all: $(OBJS)
	$(LINKER) $(OBJS) $(BLIS_LIB) -o test.x $(LDFLAGS)

%.o: %.c
	$(CC) $(CDEBUG) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) *.x