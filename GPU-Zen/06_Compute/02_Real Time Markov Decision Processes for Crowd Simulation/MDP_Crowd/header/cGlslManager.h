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
#include <sys/stat.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <vector>
#include <map>

#include "cMacros.h"
#include "cStringUtils.h"
#include "cGlErrorManager.h"
#include "cLogManager.h"

using namespace std;

//=======================================================================================

#ifndef __SHADER_OBJECT
#define __SHADER_OBJECT

class ShaderObject
{
public:
						ShaderObject					( LogManager*	log_manager_,
														  char*			vertex,
														  char*			fragment				);
						ShaderObject					( LogManager*	log_manager_,
														  char*			fragment				);
#ifdef USE_GEOMETRY_SHADERS
						ShaderObject					( LogManager*	log_manager_,
														  char*			vertex,
														  char*			fragment,
														  char*			geometry,
														  unsigned int	inPrimitive,
														  unsigned int	outPrimitive			);
						ShaderObject					( LogManager*	log_manager_,
														  char*			vertex,
														  char*			geometry,
														  unsigned int	inPrimitive,
														  unsigned int	outPrimitive			);
#endif
						~ShaderObject					( void									);

	bool				isShaderOk						( void									);
	void				setUniformi						( char*			name,
														  int			value					);
	void				setUniformf						( char*			name,
														  float			value					);
	void				setUniformiv					( char*			name,
														  int*			value,
														  int			size					);
	void				setUniformfv					( char*			name,
														  float*		value,
														  int			size					);
	void				setUniformMatrix				( char*			name,
														  float*		value,
														  int			size					);
	void				setBindableUniform				( char*			name,
														  unsigned int	buffer					);
	void				unSetBindableUniform			( char*			name					);
	void				getMaxVertexBindableUniforms	( int&			val						);
	void				getMaxGeometryBindableUniforms	( int&			val						);
	void				getMaxFragmentBindableUniforms	( int&			val						);
	void				getMaxBindableUniformSize		( int&			val						);
	int					getBindableUniformSize			( char*			name					);
	void				activate						( void									);
	static void			deactivate						( void									);
public:
	unsigned int		shader_id;
	int					num_vars;
	char*				variable_names[SHADER_OBJECT_MAX_VARIABLES];
	int					variable_ids[SHADER_OBJECT_MAX_VARIABLES];
	unsigned int		variable_types[SHADER_OBJECT_MAX_VARIABLES];
	int					variable_sizes[SHADER_OBJECT_MAX_VARIABLES];
private:
	bool				init							( char*			fragment				);
	bool				init							( char*			vertex,
														  char*			fragment				);
#ifdef USE_GEOMETRY_SHADERS
	bool				init							( char*			vertex,
														  char*			fragment,
														  char*			geometry,
														  unsigned int	input,
														  unsigned int	output					);
	bool				init							( char*			vertex,
														  char*			geometry,
														  unsigned int	inPrimitive,
														  unsigned int	outPrimitive			);
#endif
	char*				loadShaderCode					( const char*	fileName				);
	bool				initVarInfo						( void									);
	char*				registerString					( char*			str						);
private:
	LogManager*			log_manager;
	bool				shaderOK;
	char				string_utils_buffer[STRING_UTILS_BUFFER_SIZE];
	int					string_utils_buffer_ptr;
};

#endif

//=======================================================================================

#ifndef __INPUT_SHADER
#define __INPUT_SHADER

class InputShader
{
public:
									InputShader( string& name )
									{
										s_name = string( name );
										is_transform_feedback = false;
										s_ipri = 0;
										s_opri = 0;
									};
									~InputShader( void )
									{
										s_uni_i.clear();
										s_uni_f.clear();
										s_uni_iv.clear();
										s_uni_fv.clear();
										s_uni_fm.clear();
										transform_feedback_vars.clear();
									};
	string							s_name;
	string							s_vert;
	string							s_frag;
	string							s_geom;
	unsigned int					s_ipri;
	unsigned int					s_opri;
	map<string,int>					s_uni_i;
	map<string,float>				s_uni_f;
	typedef vector<int>				IV;
	map<string,IV>					s_uni_iv;
	typedef vector<float>			FV;
	map<string,FV>					s_uni_fv;
	map<string,FV>					s_uni_fm;
	bool							is_transform_feedback;
	vector<string>					transform_feedback_vars;
};

#endif

//=======================================================================================

#ifndef __GLSL_MANAGER
#define __GLSL_MANAGER

class GlslManager
{
public:
									GlslManager			(	GlErrorManager*			err_manager_,
															LogManager*				log_manager_		);
									~GlslManager		(	void										);

	bool							init				(	vector<InputShader*>&	inputs				);
	void							activate			(	string					shader_name			);
	void							deactivate			(	string					shader_name			);
	void							pause				(	void										);
	void							resume				(	void										);
	unsigned int					getId				(	string					shader_name			);
	void							setUniformfv		(	string					shader_name,
															char*					name,
															float*					val,
															int						size				);
	void							setUniformi			(	string					shader_name,
															char*					name,
															int						i					);
	void							setUniformf			(	string					shader_name,
															char*					name,
															float					f					);
	void							setUniformMatrix	(	string					shader_name,
															char*					name,
															float*					value,
															int						size				);
	bool							update				(	void										);
	bool							setup_shaders		(	void										);
private:
	typedef struct
	{
		string						name;
		ShaderObject*				shader;
		bool						active;
	}								Shader;

	map<string, int>				shaders_map;
	vector<Shader>					shaders;
	vector<Shader>					shader_stack;
	vector<ShaderObject*>			shader_objects;
	vector<InputShader*>			inputs;
private:
	void							update_shaders		(	void										);

	GlErrorManager*					err_manager;
	LogManager*						log_manager;
};

#endif

//=======================================================================================
