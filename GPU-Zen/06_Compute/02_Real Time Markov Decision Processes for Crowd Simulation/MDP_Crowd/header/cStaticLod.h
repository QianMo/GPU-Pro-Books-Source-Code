/*

Copyright 2013,2014 Sergio Ruiz, Benjamin Hernandez

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

In case you, or any of your employees or students, publish any article or
other material resulting from the use of this  software, that publication
must cite the following references:

Sergio Ruiz, Benjamin Hernandez, Adriana Alvarado, and Isaac Rudomin. 2013.
Reducing Memory Requirements for Diverse Animated Crowds. In Proceedings of
Motion on Games (MIG '13). ACM, New York, NY, USA, , Article 55 , 10 pages.
DOI: http://dx.doi.org/10.1145/2522628.2522901

Sergio Ruiz and Benjamin Hernandez. 2015. A Parallel Solver for Markov Decision Process
in Crowd Simulations. Fourteenth Mexican International Conference on Artificial
Intelligence (MICAI), Cuernavaca, 2015, pp. 107-116.
DOI: 10.1109/MICAI.2015.23

*/

#pragma once

#include "cMacros.h"
#include "cVertex.h"
#include "cVboManager.h"
#include "cGlErrorManager.h"
#include "cCamera.h"

#include <string>

using namespace std;

//=======================================================================================

#ifndef __STATIC_LOD
#define __STATIC_LOD

class StaticLod
{
public:
					StaticLod					(	GlslManager*		glsl_manager_,
													VboManager*			vbo_manager_,
													GlErrorManager*		err_manager_,
													string				owner_name_				);
					~StaticLod					(	void										);

	void			init						(	unsigned int		numAgentsWidth_,
													unsigned int		numAgentsHeight_		);
	void			runAssignment				(	unsigned int		target,
													unsigned int		posID,
													unsigned int		agentsIdsTexture,
													struct sVBOLod*		vboLOD,
													Camera*				cam						);
	void			runAssignmentCuda			(	unsigned int		target,
													unsigned int		posBufferId,
													unsigned int		agentsIdsTexture,
													struct sVBOLod*		vboLOD,
													Camera*				cam						);

	unsigned int	runSelection				(	unsigned int		target,
													float				groupID,
													struct sVBOLod*		vboLOD,
													Camera*				cam						);
	unsigned int	runSelectionCuda			(	unsigned int		target,
													float				groupID,
													struct sVBOLod*		vboLOD,
													Camera*				cam						);

	unsigned int	runAssignmentAndSelection	(	unsigned int		target,
													unsigned int		posID,
													struct sVBOLod*		vboLOD,
													Camera*				cam						);
	unsigned int	runAssignmentAndSelection	(	unsigned int		target,
													unsigned int		posID,
													unsigned int		agentsIdsTexture,
													struct sVBOLod*		vboLOD,
													Camera*				cam						);

	enum			{ VBO_CULLED };
	unsigned int	numAgentsWidth;
	unsigned int	numAgentsHeight;

	unsigned int	primitivesWritten[5];

	unsigned int	texCoords;

private:
	void			initTexCoords				(	unsigned int		target					);
	void			initTexCoordsCuda			(	unsigned int		target					);

	GlslManager*	glsl_manager;
	VboManager*		vbo_manager;
	GlErrorManager*	err_manager;

	string			owner_name;

	unsigned int	numElements;
	unsigned int	numLODs;
	unsigned int*	posVBOLODs;
	unsigned int	query_generated;
	unsigned int	query_written;
	unsigned int*	posTexBufferId;

	unsigned int	primitivesGenerated[5];
	unsigned int	lodAid;
	unsigned int	lodSid;
	int				locAid[1];
	int				locSid[1];
	unsigned int	tc_index;
	unsigned int	tc_frame;
	unsigned int	tc_size;

	string			str_tang;
	string			str_AGENTS_NPOT;
	string			str_nearPlane;
	string			str_farPlane;
	string			str_ratio;
	string			str_X;
	string			str_Y;
	string			str_Z;
	string			str_camPos;
	string			str_lod;
	string			str_groupId;
};

#endif

//=======================================================================================
