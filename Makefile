baseline:
	gcc src/baseline.c -o baseline.o -O0
	./baseline.o

clean:
	rm -f baseline.o