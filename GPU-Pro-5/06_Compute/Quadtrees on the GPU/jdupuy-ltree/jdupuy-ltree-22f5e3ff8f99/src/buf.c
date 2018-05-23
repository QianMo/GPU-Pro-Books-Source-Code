#include "buf.h"

void buf_do_realloc_(void **a, int nr, int sz) {
	int *p = realloc(*a ? buf_raw_(*a) : NULL, 2 * sizeof(int) + nr * sz);
	p[0] = nr;
	if (!*a)
		p[1] = 0;
	*a = p + 2;
}

