#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  if (argc != 1) {
      fprintf(stderr, "Usage: print\n");
      exit(1);
  }
  printf("%-20s%-7s%-7s%-7s%-5s%-7s\n", "Implementation", "M", "N", "K", "NP", "Time");
  printf("-----------------------------------------------------------\n");
  return 0;
}
