OBJS=getfeatures.o
CC= g++ -lm
CFLAGS=-lm -c -Wall -g 

getfeatures:$(OBJS)
	$(CC)  $^ -o getfeatures
%.o:%.c
	$(CC)  $^ $(CFLAGS)  -o $@

clean:
	$(RM) *.o getfeatures -r
