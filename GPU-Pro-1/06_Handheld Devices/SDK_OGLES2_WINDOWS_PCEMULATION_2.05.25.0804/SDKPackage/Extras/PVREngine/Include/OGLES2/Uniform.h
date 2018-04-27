/******************************************************************************

 @File         Uniform.h

 @Title        A class for holding information about shader uniforms as they are loaded.

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent - OGL/OGLES/OGLES2 Specific at the moment

 @Description  A class for holding information about shader uniforms as they are
               loaded.

******************************************************************************/
#ifndef UNIFORM_H
#define UNIFORM_H

#include "../PVRTools.h"
#include "Semantics.h"

namespace pvrengine
{
	/*!***************************************************************************
	* @Class Uniform
	* @Brief 	A class for holding information about shader uniforms.
	* @Description 	A class for holding information about shader uniforms.
	*****************************************************************************/
	class Uniform
	{
	public:
		/*!****************************************************************************
		@Function		Uniform
		@Description	Blank constructor from PFX
		******************************************************************************/
		Uniform(){}

		/*!****************************************************************************
		@Function		Uniform
		@Input			sUniform - a PFX uniform struct
		@Description	Constructor from PFX uniform
		******************************************************************************/
		Uniform(const SPVRTPFXUniform sUniform):
		m_u32Location(sUniform.nLocation),
			m_eSemantic((EUniformSemantic)sUniform.nSemantic),
			m_u32Idx(sUniform.nIdx)
		{}

		/*!****************************************************************************
		@Function		getSemantic
		@Return			the enum identifier of this uniform
		@Description	Accessor for the enum identifier of this uniform
		******************************************************************************/
		EUniformSemantic getSemantic() const{return m_eSemantic;}

		/*!****************************************************************************
		@Function		getLocation
		@Return			the shader location for this uniform
		@Description	Accessor for the location for this uniform
		******************************************************************************/
		unsigned int getLocation() const{return m_u32Location;}

		/*!****************************************************************************
		@Function		getIdx
		@Return			the index of this uniform
		@Description	Accessor for the index of this uniform
		******************************************************************************/
		unsigned int getIdx() const	{return m_u32Idx;}

	private:
		unsigned int	m_u32Location;		/*! GL location of the Uniform */
		EUniformSemantic	m_eSemantic;	/*! Application-defined semantic value */
		unsigned int	m_u32Idx;			/*! Index; for example two semantics might be LIGHTPOSITION0 and LIGHTPOSITION1 */

	};
}


#endif // UNIFORM_H

/******************************************************************************
End of file (Uniform.h)
******************************************************************************/
