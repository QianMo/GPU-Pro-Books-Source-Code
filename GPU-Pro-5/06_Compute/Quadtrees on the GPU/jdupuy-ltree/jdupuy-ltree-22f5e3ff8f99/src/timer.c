#include <stdlib.h>
#include <stdio.h>
#include "glload.h"
#include "timer.h"

struct timer_t {
	GLdouble cpu_ticks, gpu_ticks;
	GLint64 cpu_start_ticks;
	GLuint queries[2];
	GLint is_gpu_ticking,
	      is_cpu_ticking,
	      is_gpu_ready;
};

/*
create
*/
struct timer_t *timer_create(void) {
	struct timer_t *timer = malloc(sizeof(*timer));

	if(!timer) {
		fprintf(stderr, "timer: malloc failed\n");
		return NULL;
	}
	glGenQueries(2,timer->queries);
	timer->cpu_ticks = 0.0;
	timer->gpu_ticks = 0.0;
	timer->cpu_start_ticks = 0.0;
	timer->is_cpu_ticking = GL_FALSE;
	timer->is_gpu_ticking = GL_FALSE;
	timer->is_gpu_ready = GL_TRUE;
	glQueryCounter(timer->queries[0], GL_TIMESTAMP);
	glQueryCounter(timer->queries[1], GL_TIMESTAMP);
	return timer;
}

/*
release
*/
void timer_release(struct timer_t *timer) {
	if(timer) {
		glDeleteQueries(2,timer->queries);
		free(timer);
		timer = NULL;
	}
}

/*
start
*/
void timer_start(struct timer_t *timer) {
	if(!timer->is_cpu_ticking) {
		timer->is_cpu_ticking = GL_TRUE;
		glGetInteger64v(GL_TIMESTAMP, &timer->cpu_start_ticks);
	}
	if(!timer->is_gpu_ticking && timer->is_gpu_ready) {
		glQueryCounter(timer->queries[0], GL_TIMESTAMP);
		timer->is_gpu_ticking = GL_TRUE;
	}
}

/*
stop
*/
void timer_stop(struct timer_t *timer) {
	if(timer->is_cpu_ticking) {
		GLint64 now = 0;

		glGetInteger64v(GL_TIMESTAMP, &now);
		timer->cpu_ticks = (now - timer->cpu_start_ticks) / 1e9;
		timer->is_cpu_ticking = GL_FALSE;
	}
	if(timer->is_gpu_ticking && timer->is_gpu_ready) {
		glQueryCounter(timer->queries[1], GL_TIMESTAMP);
		timer->is_gpu_ticking = GL_FALSE;
	}
}

/*
ticks
*/
void timer_ticks(struct timer_t *timer, double *cpu, double *gpu) {
	// return 0 if timer has not been stopped
	if(timer->is_cpu_ticking) {
		if(cpu) *cpu = 0.0;
		if(gpu) *gpu = 0.0;
		return;
	}

	// lazy gpu evaluation
	if(!timer->is_gpu_ticking) {
		glGetQueryObjectiv(timer->queries[1],
		                   GL_QUERY_RESULT_AVAILABLE,
		                   &timer->is_gpu_ready);
		if(timer->is_gpu_ready) {
			GLuint64 start, stop;

			glGetQueryObjectui64v(timer->queries[0],
			                      GL_QUERY_RESULT,
			                      &start);
			glGetQueryObjectui64v(timer->queries[1],
			                      GL_QUERY_RESULT,
			                      &stop);
			timer->gpu_ticks = (stop - start) / 1e9;
		}
	}
	if(gpu) *gpu = timer->gpu_ticks;
	if(cpu) *cpu = timer->cpu_ticks;
}



