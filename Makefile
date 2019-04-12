all:ai_tools network
	gcc -c main.c -o main.o
	gcc main.o ai_tools.o network.o -o rl.out

ai_tools:
	gcc -c ai_tools.c -o ai_tools.o

network:
	gcc -c network.c -o network.o

clean:
	rm *.o
