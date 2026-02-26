CC = gcc
CFLAGS = -Wall -O2 -g

all: tq

tq: tq.c cli.c
	$(CC) $(CFLAGS) -o tq tq.c cli.c -lm

clean:
	rm -f tq *.o
