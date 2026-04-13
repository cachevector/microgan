CC = gcc
CFLAGS = -Wall -O3 -fPIC -I.
LDFLAGS = -shared

all: libmicrogan.so

libmicrogan.so: MicroGAN_runtime.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f libmicrogan.so
