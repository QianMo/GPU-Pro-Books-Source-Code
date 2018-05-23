// from http://github.com/skaslev/catmull-clark/blob/master/buf.h
#ifndef BUF_H
#define BUF_H

#include <stdlib.h>

/* Based on Sean Barrett's stretchy buffer at http://www.nothings.org/stb/stretchy_buffer.txt
* init: NULL, free: buf_free(), push_back: buf_push(), size: buf_len()
*/
#define buf_len(a) ((a) ? buf_n_(a) : 0)
#define buf_push(a, v) (buf_maybegrow1_(a), (a)[buf_n_(a)++] = (v))
#define buf_resize(a, n) (buf_maybegrow_(a, n), (a) ? buf_n_(a) = (n) : 0)
#define buf_reserve(a, n) (buf_maybegrow_(a, n))
#define buf_free(a) ((a) ? free(buf_raw_(a)) : (void) 0)
#define buf_foreach(it, a) for ((it) = (a); (it) < (a) + buf_len(a); (it)++)
#define buf_back(a) (buf_n_(a) > 1 ? (a) + buf_n_(a) - 1 : (a))

/* Private */
#define buf_raw_(a) ((int *) (a) - 2)
#define buf_m_(a) (buf_raw_(a)[0])
#define buf_n_(a) (buf_raw_(a)[1])

#define buf_maybegrow_(a, n) (((n) > 0) && (!(a) || (n) >= buf_m_(a)) ? buf_realloc_(a, n) : (void) 0)
#define buf_maybegrow1_(a) (!(a) || buf_m_(a) == 0 ? buf_realloc_(a, 8) : \
buf_n_(a) == buf_m_(a) ? buf_realloc_(a, 2 * buf_m_(a)) : (void) 0)
#define buf_realloc_(a, n) buf_do_realloc_((void **) &(a), n, sizeof(*(a)))

void buf_do_realloc_(void **a, int nr, int sz);

#endif
