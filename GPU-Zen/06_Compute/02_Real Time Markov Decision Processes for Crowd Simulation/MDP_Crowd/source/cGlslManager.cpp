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
#include "cGlslManager.h"

//=======================================================================================
//
GlslManager::GlslManager( 	GlErrorManager*   	err_manager_,
							LogManager*			log_manager_	)
{
	err_manager	= err_manager_;
	log_manager	= log_manager_;
}
//
//=======================================================================================
//
bool GlslManager::update( void )
{
	return setup_shaders();
}
//
//=======================================================================================
//
GlslManager::~GlslManager( void )
{
	shader_objects.clear();
	shaders_map.clear();
	shaders.clear();
	shader_stack.clear();
}
//
//=======================================================================================
//
bool GlslManager::init( vector<InputShader*>&	inputs )
{
	log_manager->log( LogManager::GLSL_MANAGER, "Compiling shaders..." );
	for( unsigned int i = 0; i < inputs.size(); i++ )
	{
		GlslManager::inputs.push_back( inputs[i] );
		Shader			sIn;
		ShaderObject*	so;
		bool			is_vert = false;
		bool			is_frag = false;
		bool			is_geom = false;
		if( inputs[i]->s_vert.size() > 0 )
		{
			is_vert = true;
		}
		if( inputs[i]->s_frag.size() > 0 )
		{
			is_frag = true;
		}
		if( inputs[i]->s_geom.size() > 0 )
		{
			is_geom = true;
		}

		if( is_vert && is_frag && !is_geom )
		{
			so = new ShaderObject( 	log_manager,
									(char*)inputs[i]->s_vert.c_str(),
									(char*)inputs[i]->s_frag.c_str() );
		}
		else if( is_frag && !is_vert && !is_geom )
		{
			so = new ShaderObject( 	log_manager,
									(char*)inputs[i]->s_frag.c_str() );
		}
		else if( is_frag && is_vert && is_geom )
		{
			so = new ShaderObject( 	log_manager,
									(char*)inputs[i]->s_vert.c_str(),
									(char*)inputs[i]->s_frag.c_str(),
									(char*)inputs[i]->s_geom.c_str(),
									inputs[i]->s_ipri,
									inputs[i]->s_opri		);
		}
		else if( is_vert && is_geom )
		{
			so = new ShaderObject( 	log_manager,
									(char*)inputs[i]->s_vert.c_str(),
									(char*)inputs[i]->s_geom.c_str(),
									inputs[i]->s_ipri,
									inputs[i]->s_opri		);
		}

		if( !so->isShaderOk() )
		{
			log_manager->log( LogManager::LERROR, "While compiling shader \"%s\"!", inputs[i]->s_name.c_str() );
			return false;
		}
		sIn.name				= inputs[i]->s_name;
		sIn.shader				= so;
		sIn.active				= false;
		shaders_map[sIn.name]	= shaders.size();
		shaders.push_back( sIn );
	}

	err_manager->getError( "After instancing shader objects:" );
	if( setup_shaders() == false )
	{
		return false;
	}

	log_manager->log( LogManager::GLSL_MANAGER, "Done compiling. %u shaders linked.", (unsigned int)shaders.size() );
	return true;
}
//
//=======================================================================================
//
void GlslManager::activate( string shader_name )
{
	if( shaders_map.find( shader_name ) != shaders_map.end() )
	{
		int s = shaders_map[shader_name];
		shaders[s].active = true;
		shader_stack.push_back( shaders[s] );
		glUseProgram( shaders[s].shader->shader_id );
	}
	else
	{
		log_manager->log( LogManager::LERROR, "Shader \"%s\" not registered in "
											  "GLSL Manager.",
											  shader_name.c_str()				);
	}
}
//
//=======================================================================================
//
void GlslManager::deactivate( string shader_name )
{
	bool error = false;

	if( shaders_map.find( shader_name ) != shaders_map.end() )
	{
		int s = shaders_map[shader_name];
		shaders[s].active = false;
		if( shader_stack.size() > 0 )
		{
			shader_stack.pop_back();
		}
	}
	else
	{
		error = true;
		log_manager->log( LogManager::LERROR, "Shader \"%s\" not registered in "
											  "GLSL Manager.",
											  shader_name.c_str()				);
	}
	if( !error )
	{
		if( shader_stack.size() > 0 )
		{
			Shader temp = (Shader)shader_stack.back();
			shader_stack.pop_back();
			glUseProgram( temp.shader->shader_id );
		}
		else
		{
			glUseProgram( 0 );
		}
	}
}
//
//=======================================================================================
//
void GlslManager::setUniformfv( string shader_name, char* name, float* val, int size )
{
	if( shaders_map.find( shader_name ) != shaders_map.end() )
	{
		int s = shaders_map[shader_name];
		shaders[s].shader->setUniformfv( name, val, size );
	}
	else
	{
		log_manager->log( LogManager::LERROR, "Shader \"%s\" not registered in "
											  "GLSL Manager.",
											  shader_name.c_str()				);
	}
}
//
//=======================================================================================
//
void GlslManager::setUniformi( string shader_name, char* name, int i )
{
	if( shaders_map.find( shader_name ) != shaders_map.end() )
	{
		int s = shaders_map[shader_name];
		shaders[s].shader->setUniformi( name, i );
	}
	else
	{
		log_manager->log( LogManager::LERROR, "Shader \"%s\" not registered in "
											  "GLSL Manager.",
											  shader_name.c_str()				);
	}
}
//
//=======================================================================================
//
void GlslManager::setUniformf( string shader_name, char* name, float f )
{
	if( shaders_map.find( shader_name ) != shaders_map.end() )
	{
		int s = shaders_map[shader_name];
		shaders[s].shader->setUniformf( name, f );
	}
	else
	{
		log_manager->log( LogManager::LERROR, "Shader \"%s\" not registered in "
											  "GLSL Manager.",
											  shader_name.c_str()				);
	}
}
//
//=======================================================================================
//
void GlslManager::setUniformMatrix( string	shader_name,
									  char*		name,
									  float*	value,
									  int		size
									)
{
	if( shaders_map.find( shader_name ) != shaders_map.end() )
	{
		int s = shaders_map[shader_name];
		shaders[s].shader->setUniformMatrix( name, value, size );
	}
	else
	{
		log_manager->log( LogManager::LERROR, "Shader \"%s\" not registered in "
											  "GLSL Manager.",
											  shader_name.c_str()				);
	}
}
//
//=======================================================================================
//
unsigned int GlslManager::getId( string shader_name )
{
	if( shaders_map.find( shader_name ) != shaders_map.end() )
	{
		int s = shaders_map[shader_name];
		return shaders[s].shader->shader_id;
	}
	else
	{
		log_manager->log( LogManager::LERROR, "Shader \"%s\" not registered in "
											  "GLSL Manager.",
											  shader_name.c_str()				);
		return 0;
	}
}
//
//=======================================================================================
//
void GlslManager::pause( void )
{
	glUseProgram( 0 );
}
//
//=======================================================================================
//
void GlslManager::resume( void )
{
	if( shader_stack.size() > 0 )
	{
		Shader temp = (Shader)shader_stack.back();
		glUseProgram( temp.shader->shader_id );
	}
}
//
//=======================================================================================
//
bool GlslManager::setup_shaders( void )
{
	for( unsigned int s = 0; s < shaders.size(); s++ )
	{
		shaders[s].shader->activate();
		{
			if( inputs[s]->is_transform_feedback )
			{
				for( unsigned int tfv = 0; tfv < inputs[s]->transform_feedback_vars.size(); tfv++ )
				{
					GLint loc[1];
					loc[0] = glGetVaryingLocationNV( shaders[s].shader->shader_id,
													 inputs[s]->transform_feedback_vars[tfv].c_str() );
					glTransformFeedbackVaryingsNV( shaders[s].shader->shader_id, 1, loc, GL_SEPARATE_ATTRIBS_NV );
				}
				glLinkProgram( shaders[s].shader->shader_id );
			}

			map<string,int>::iterator it_ui;
			int ui_count = 0;
			for( it_ui = inputs[s]->s_uni_i.begin(); it_ui != inputs[s]->s_uni_i.end(); it_ui++ )
			{
				ui_count++;
				// If count is greater than 1 and the indicated uniform variable is not an array,
				// a GL_INVALID_OPERATION error is generated and the specified uniform variable
				// will remain unchanged.
				// Source: http://www.opengl.org/sdk/docs/man/xhtml/glUniform.xml
				if( ui_count > 2 )
				{
					int val[1] = { (*it_ui).second };
					shaders[s].shader->setUniformiv( (char*)(*it_ui).first.c_str(), &val[0], 1 );
				}
				else
				{
					shaders[s].shader->setUniformi( (char*)(*it_ui).first.c_str(), (*it_ui).second );
				}
			}

			map<string,float>::iterator it_uf;
			for ( it_uf = inputs[s]->s_uni_f.begin(); it_uf != inputs[s]->s_uni_f.end(); it_uf++ )
			{
				shaders[s].shader->setUniformf( (char*)(*it_uf).first.c_str(), (*it_uf).second );
			}

			map<string,InputShader::IV>::iterator it_uiv;
			for ( it_uiv = inputs[s]->s_uni_iv.begin(); it_uiv != inputs[s]->s_uni_iv.end(); it_uiv++ )
			{
				shaders[s].shader->setUniformiv( (char*)(*it_uiv).first.c_str(), &((*it_uiv).second)[0], (*it_uiv).second.size() );
			}

			map<string,InputShader::FV>::iterator it_ufv;
			for ( it_ufv = inputs[s]->s_uni_fv.begin(); it_ufv != inputs[s]->s_uni_fv.end(); it_ufv++ )
			{
				shaders[s].shader->setUniformfv( (char*)(*it_ufv).first.c_str(), &((*it_ufv).second)[0], (*it_ufv).second.size() );
			}

			map<string,InputShader::FV>::iterator it_ufm;
			for ( it_ufm = inputs[s]->s_uni_fm.begin(); it_ufm != inputs[s]->s_uni_fm.end(); it_ufm++ )
			{
				shaders[s].shader->setUniformMatrix( (char*)(*it_ufm).first.c_str(), &((*it_ufm).second)[0], (*it_ufm).second.size() );
			}
		}
		shaders[s].shader->deactivate();
		char msg[100];
		sprintf( msg, "After setting up shader \"%s\".", shaders[s].name.c_str() );
		if( err_manager->getError( msg ) != GL_NO_ERROR )
		{
			return false;
		}
	}
	return true;
}
//
//=======================================================================================
