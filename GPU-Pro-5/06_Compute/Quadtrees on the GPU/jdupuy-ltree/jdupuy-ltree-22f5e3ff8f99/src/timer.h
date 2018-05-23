#ifndef TIMER_H
#define TIMER_H

// create / release
struct timer_t *timer_create(void);
void timer_release(struct timer_t *timer);

// manipulation
void timer_start(struct timer_t *timer);
void timer_stop(struct timer_t *timer);
void timer_ticks(struct timer_t *timer, double *cpu, double *gpu);

#endif
