#ifndef __SHARED_TTYPES__
#define __SHARED_TTYPES__

#define GammatonFlag_Alive 1
#define GammatonFlag_Hit 2
#define GammatonFlag_Bounce 4

#define SET_FLAG(variable, flag) ((variable) = (variable) | flag)
#define CLEAR_FLAG(variable, flag) ((variable) = (variable) - ((variable) & flag))
#define HAS_FLAG(variable, flag) (((variable) & flag)==flag)

#define SET_ALIVE(variable) (SET_FLAG((variable), GammatonFlag_Alive))
#define SET_DEAD(variable) (CLEAR_FLAG((variable), GammatonFlag_Alive))
#define IS_ALIVE(variable) (HAS_FLAG((variable), GammatonFlag_Alive))
#define IS_DEAD(variable) (!HAS_FLAG((variable), GammatonFlag_Alive))

#define SET_HIT(variable) (SET_FLAG((variable), GammatonFlag_Hit))
#define SET_MIDAIR(variable) (CLEAR_FLAG((variable), GammatonFlag_Hit))
#define IS_HIT(variable) (HAS_FLAG((variable), GammatonFlag_Hit))
#define IS_MIDAIR(variable) (!HAS_FLAG((variable), GammatonFlag_Hit))

#define SET_BOUNCE(variable) (SET_FLAG((variable), GammatonFlag_Bounce))
#define SET_FLOAT(variable) (CLEAR_FLAG((variable), GammatonFlag_Bounce))
#define IS_BOUNCE(variable) (HAS_FLAG((variable), GammatonFlag_Bounce))
#define IS_FLOAT(variable) (!HAS_FLAG((variable), GammatonFlag_Bounce))

struct MaterialProperty
{
	float Water;
	float Dirt;
	float Metal;
	float Wood;
	float Organic;
	float Rust;
	float Stone;
	float Dummy;
};

#endif
