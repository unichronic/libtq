CC = gcc
CFLAGS = -Wall -O2 -g

all: tq test_tq

tq: tq.c cli.c
	$(CC) $(CFLAGS) -o tq tq.c cli.c -lm

test_tq: tq.c test_tq.c
	$(CC) $(CFLAGS) -o test_tq tq.c test_tq.c -lm

clean:
	rm -f tq test_tq *.bin *.tqb
