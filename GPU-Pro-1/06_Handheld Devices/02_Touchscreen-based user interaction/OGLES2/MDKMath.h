/******************************************************************************

 @File         MDKMath.h

 @Title        MDKTools

 @Copyright    Copyright (C) 2009 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  Math utility functions
 
******************************************************************************/


#ifndef _MDK_MATH_H_
#define _MDK_MATH_H_

#include <math.h>

namespace MDK {
	namespace Math {

		/*
		 * Exp and DerivativeExp provide two functions designed on purpose for objects in motion.
		 * Sample plots:
		 * y |
		 * A -                  ----------
		 *   |                /
		 *   |              /
		 *   |            /
		 *   |           /
		 *   +==========*------------------->
		 *              0                   t
		 *
		 * y |
		 * A -          |
		 *   |          |\
		 *   |          | \
		 *   |          |  \
		 *   |          |    \ 
		 *   +==========*     ==============>
		 *              0                   t
		 * DerivativeExp is better used in integral functions.
		 * 
		 */

		
		inline float Exp(float time, float amplitude, float tau)
		{
			if (time < -0.001f)
				return 0.0f;

			return amplitude * (1.0f - exp(-tau * time));
		}


		inline float DerivativeExp(float time, float amplitude, float tau)
		{
			if (time < -0.001f)
				return 0.0f;

			return tau * amplitude * exp(-tau * time);
		}
	}
}

#endif
