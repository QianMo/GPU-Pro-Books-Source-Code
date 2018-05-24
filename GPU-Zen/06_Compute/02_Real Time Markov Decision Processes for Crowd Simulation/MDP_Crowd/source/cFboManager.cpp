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

#include "cFboManager.h"

/*
	NOTE: When using texture for DEPTH lookup, GL_TEXTURE_COMPARE_MODE = GL_COMPARE_R_TO_TEXTURE
	      When using texture for COLOR lookup, GL_TEXTURE_COMPARE_MODE = GL_NONE
	Source: http://www.gpgpu.org/forums/viewtopic.php?p=19292&sid=469a1cf984dc080554731ad3811f4ef4
*/

//============================================================================================================
//
FboManager::FboManager( LogManager* log_manager_, GlslManager* glsl_manager_, vector<InputFbo*>& inputs )
{
	this->inputs			= inputs;
	fboOK					= false;
	log_manager				= log_manager_;
	this->err_manager		= new GlErrorManager( log_manager );
	glsl_manager			= glsl_manager_;
	proj_manager			= new ProjectionManager();
	screen_text				= new ScreenText();
	// Init DevIL:
	//ilInit();
	//ilEnable( IL_ORIGIN_SET );
	// Query for how many attachments a FBO can handle:
	glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS_EXT, &maxAttachments );
	log_manager->log( LogManager::FBO_MANAGER, "Max Color Attachments: %i", maxAttachments );
	// Check for errors before we begin configuring:
	err_manager->getError( "Before configuring" );
}
//
//============================================================================================================
//
FboManager::~FboManager( void )
{
	FREE_INSTANCE( proj_manager );
	FREE_INSTANCE( err_manager  );
	FREE_INSTANCE( screen_text  );
	fbos.clear();
	texture_ids.clear();
	targets_map.clear();
	tex_fbo_map.clear();
	inputs.clear();
}
//
//============================================================================================================
//
void FboManager::displayTexture( string			shader_name,
								 string			tex_name,
								 unsigned int	quad_width,
								 unsigned int	quad_height,
								 unsigned int	texture_width,
								 unsigned int	texture_height
							   )
{
	glPushAttrib( GL_LIGHTING_BIT | GL_DEPTH_BUFFER_BIT );
	{
		glDisable( GL_LIGHTING );
		glDisable( GL_DEPTH_TEST );
		proj_manager->setOrthoProjection( texture_width, texture_height );
		{
			pushActiveBind( tex_name, 0 );
			{
				renderQuad( shader_name,
							quad_width,
							quad_height,
							texture_width,
							texture_height );
			}
			popActiveBind();
		}
		proj_manager->restoreProjection();
	}
	glPopAttrib();
}
//
//============================================================================================================
//
void FboManager::displayTexture( 	string 			shader_name,
									string 			tex_name,
									unsigned int 	width,
									unsigned int 	height		)
{
	glPushAttrib( GL_LIGHTING_BIT | GL_DEPTH_BUFFER_BIT );
	{
		glDisable( GL_LIGHTING );
		glDisable( GL_DEPTH_TEST );
		proj_manager->setOrthoProjection( width, height, false );
		{
			pushActiveBind( tex_name, 0 );
			{
				renderQuad( tex_fbo_map[tex_name].fbo_name, (char*)shader_name.c_str(), width, height );
			}
			popActiveBind();
		}
		proj_manager->restoreProjection();
	}
	glPopAttrib();
}
//
//============================================================================================================
//
void FboManager::drawQuad( unsigned int w, unsigned int h, unsigned int tw, unsigned int th )
{
	glBegin( GL_QUADS );
	{
		glTexCoord2f( 0.0f,           0.0f ); glVertex2f( 0.0f,     0.0f      );
		glTexCoord2f( (float)tw,      0.0f ); glVertex2f( (float)w, 0.0f      );
		glTexCoord2f( (float)tw, (float)th ); glVertex2f( (float)w, (float)h  );
		glTexCoord2f( 0.0f,      (float)th ); glVertex2f( 0.0f,     (float)h  );
	}
	glEnd();
}
//
//============================================================================================================
//
void FboManager::drawQuadArray( unsigned int w, unsigned int h, unsigned int tw, unsigned int th )
{
	glBegin( GL_QUADS );
	{
		glTexCoord3f( 0.0f,      0.0f,      0.0f ); glVertex2f( 0,               0 );
		glTexCoord3f( (float)tw, 0.0f,      0.0f ); glVertex2f( (float)w,        0 );
		glTexCoord3f( (float)tw, (float)th, 0.0f ); glVertex2f( (float)w, (float)h );
		glTexCoord3f( 0.0f,      (float)th, 0.0f ); glVertex2f( 0,        (float)h );
	}
	glEnd();
}
//
//============================================================================================================
//
void FboManager::renderQuad( string			fbo_name,
							 string			shader_name,
						     unsigned int	width,
						     unsigned int	height		)
{
	glsl_manager->activate( shader_name );
	{
		if( fbos[fbo_name].fbo_tex_target == GL_TEXTURE_2D )
		{
			drawQuad( width, height, 1, 1 );
		}
		else
		{
			drawQuad( width, height, fbos[fbo_name].fbo_width, fbos[fbo_name].fbo_height );
		}
	}
	glsl_manager->deactivate( shader_name );
}
//
//============================================================================================================
//
void FboManager::renderQuad( string			shader_name,
						     unsigned int	quad_width,
						     unsigned int	quad_height,
							 unsigned int	texture_width,
							 unsigned int	texture_height
						   )
{
	glsl_manager->activate( shader_name );
	{
		drawQuad( quad_width, quad_height, texture_width, texture_height );
	}
	glsl_manager->deactivate( shader_name );
}
//
//============================================================================================================
//
void FboManager::renderTiled( string		shader_name,
							  string		tex_name,
							  unsigned int	tile_number,
							  unsigned int	quad_width,
							  unsigned int	quad_height,
							  unsigned int	texture_width,
							  unsigned int	texture_height
							)
{
	glPushAttrib( GL_LIGHTING_BIT | GL_DEPTH_BUFFER_BIT );
	{
		glDisable( GL_LIGHTING );
		glDisable( GL_DEPTH_TEST );
		proj_manager->setOrthoProjection( quad_width * tile_number / 4,
										  0,
										  quad_width  / 4,
										  quad_height / 4,
										  0,
										  quad_width,
										  0,
										  quad_height,
										  true							);
		pushActiveBind( tex_name, 0 );
		{
			renderQuad( shader_name,
						quad_width,
						quad_height,
						texture_width,
						texture_height );
		}
		popActiveBind();
		proj_manager->restoreProjection();
	}
	glPopAttrib();
}
//
//============================================================================================================
//
const char* FboManager::isFboValid( unsigned int fb )
{
	// Fuente: http://www.gamedev.net/community/forums/topic.asp?topic_id=364174
	string m_sLastError( "FBO_OK" );
	GLenum status;
	glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, fb );
	status                = glCheckFramebufferStatusEXT( GL_FRAMEBUFFER_EXT );
	switch( status )
	{
		case GL_FRAMEBUFFER_COMPLETE_EXT:
			fboOK         = true;
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
			m_sLastError = string( "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT" );
			fboOK         = false;
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
			m_sLastError = string( "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT" );
			fboOK         = false;
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
			m_sLastError = string( "GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT" );
			fboOK         = false;
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
			m_sLastError = string( "GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT" );
			fboOK         = false;
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
			m_sLastError = string( "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT" );
			fboOK         = false;
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
			m_sLastError = string( "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT" );
			fboOK         = false;
			break;
		case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
			m_sLastError = string( "GL_FRAMEBUFFER_UNSUPPORTED_EXT" );
			fboOK         = false;
			break;
		default:
			m_sLastError = string( "UNKNOWN_ERROR" );
			fboOK         = false;
			break;
	}
	return( (char*)m_sLastError.c_str() );
}
//
//============================================================================================================
//
void FboManager::typicalTextureSettings( char*			msg,
										 unsigned int	tex_target,
										 unsigned int&	texture,
										 unsigned int	w,
										 unsigned int	h
									   )
{
	glGenTextures	( 1,		  &texture									);
	glBindTexture	( tex_target, texture									);
	glTexImage2D	( tex_target,
					  0,
					  GL_RGBA,
					  w,
					  h,
					  0,
					  GL_RGBA,
					  GL_FLOAT,
					  NULL		);
	glTexParameteri	( tex_target, GL_TEXTURE_MIN_FILTER, GL_LINEAR			);
	glTexParameteri	( tex_target, GL_TEXTURE_MAG_FILTER, GL_LINEAR			);
	glTexParameteri	( tex_target, GL_TEXTURE_WRAP_S,	 GL_CLAMP_TO_EDGE	);
	glTexParameteri	( tex_target, GL_TEXTURE_WRAP_T,	 GL_CLAMP_TO_EDGE	);
	err_manager->getError( msg );
}
//
//============================================================================================================
//
void FboManager::gpgpuTextureSettings( char*			msg,
									   unsigned int		tex_target,
									   unsigned int&	texture,
									   unsigned int		w,
									   unsigned int		h
									 )
{
	glGenTextures	( 1, &texture																		);
	glBindTexture	( tex_target, texture																);
	glTexImage2D	( tex_target, 0,					 GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, NULL	);
	glTexParameteri	( tex_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST										);
	glTexParameteri	( tex_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST										);
	glTexParameteri	( tex_target, GL_TEXTURE_WRAP_S,     GL_CLAMP										);
	glTexParameteri	( tex_target, GL_TEXTURE_WRAP_T,     GL_CLAMP										);
	err_manager->getError( msg );
}
//
//============================================================================================================
//
void FboManager::depthTextureNoCompareSettings( char*			msg,
											    unsigned int	tex_target,
											    unsigned int&	depthTexture,
											    unsigned int	w,
											    unsigned int	h
											  )
{
	// Spec: http://www.opengl.org/registry/specs/EXT/packed_depth_stencil.txt
	// Source: http://onesadcookie.com/svn/ARB_draw_buffers_Test/main.c
	glGenTextures	( 1, &depthTexture );
	glBindTexture	( tex_target, depthTexture );
	glTexImage2D	( tex_target,
					  0,
#if defined( BUGGY_PACKED_DEPTH_STENCIL )
					  GL_DEPTH_COMPONENT24,
#else
					  GL_DEPTH24_STENCIL8_EXT,
#endif
					  w,
					  h,
					  0,
#if defined( BUGGY_PACKED_DEPTH_STENCIL )
					  GL_DEPTH_COMPONENT,
					  GL_UNSIGNED_BYTE,
#else
					  GL_DEPTH_STENCIL_EXT,
					  GL_UNSIGNED_INT_24_8_EXT,
#endif
					  NULL						);

	glTexParameteri	( tex_target, GL_TEXTURE_COMPARE_MODE, GL_NONE          );
	glTexParameteri	( tex_target, GL_TEXTURE_MIN_FILTER,   GL_LINEAR        );
	glTexParameteri	( tex_target, GL_TEXTURE_MAG_FILTER,   GL_LINEAR        );
	glTexParameteri	( tex_target, GL_TEXTURE_WRAP_S,       GL_CLAMP_TO_EDGE );
	glTexParameteri	( tex_target, GL_TEXTURE_WRAP_T,       GL_CLAMP_TO_EDGE );
	err_manager->getError( msg );
}
//
//============================================================================================================
//
void FboManager::depthTextureRCompareSettings( char*			msg,
											   unsigned int		tex_target,
											   unsigned int&	depthTexture,
											   unsigned int		w,
											   unsigned int		h
										     )
{
	// Spec: http://www.opengl.org/registry/specs/EXT/packed_depth_stencil.txt
	// Source: http://onesadcookie.com/svn/ARB_draw_buffers_Test/main.c
	glGenTextures	( 1, &depthTexture );
	glBindTexture	( tex_target, depthTexture );
	glTexImage2D	( tex_target,
					  0,
#if defined( BUGGY_PACKED_DEPTH_STENCIL )
					  GL_DEPTH_COMPONENT24,
#else
					  GL_DEPTH24_STENCIL8_EXT,
#endif
					  w,
					  h,
					  0,
#if defined( BUGGY_PACKED_DEPTH_STENCIL )
					  GL_DEPTH_COMPONENT,
					  GL_UNSIGNED_BYTE,
#else
					  GL_DEPTH_STENCIL_EXT,
					  GL_UNSIGNED_INT_24_8_EXT,
#endif
					  NULL );

	glTexParameteri	( tex_target, GL_TEXTURE_MIN_FILTER,   GL_LINEAR               );
	glTexParameteri	( tex_target, GL_TEXTURE_MAG_FILTER,   GL_LINEAR               );
	glTexParameteri	( tex_target, GL_TEXTURE_WRAP_S,       GL_CLAMP_TO_EDGE        );
	glTexParameteri	( tex_target, GL_TEXTURE_WRAP_T,       GL_CLAMP_TO_EDGE        );
	glTexParameteri	( tex_target, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE );
	glTexParameteri	( tex_target, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL               );
	glTexParameteri	( tex_target, GL_DEPTH_TEXTURE_MODE,   GL_INTENSITY            );
	err_manager->getError( msg );
}
//
//============================================================================================================
//
void FboManager::registerColorTarget( string		target_name,
									  unsigned int	tex_target,
									  unsigned int	target_color,
									  unsigned int& target_texture
									)
{
	unsigned int targetColor = GL_COLOR_ATTACHMENT0_EXT + target_color;
	glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT,
							   targetColor,
							   tex_target,
							   target_texture,
							   0/*mipmap level*/);
	targets_map[target_name] = targetColor;
}
//
//============================================================================================================
//
void FboManager::registerDepthTarget( unsigned int	tex_target,
									  unsigned int& target_texture
									)
{
	glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT,
							   GL_DEPTH_ATTACHMENT_EXT,
							   tex_target,
							   target_texture,
							   0/*mipmap level*/);
#if !defined( BUGGY_PACKED_DEPTH_STENCIL )
	glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT,
							   GL_STENCIL_ATTACHMENT_EXT,
							   tex_target,
							   target_texture,
							   0/*mipmap level*/);
#endif
}
//
//============================================================================================================
//
void FboManager::setTarget( string	tname,
						    bool	clear_color_bit,
							bool	clear_depth_bit
						  )
{
	if( targets_map.find( tname ) != targets_map.end() )
	{
		glDrawBuffer( targets_map[tname] );
		if( clear_color_bit )
		{
			glClear( GL_COLOR_BUFFER_BIT );
		}
		if( clear_depth_bit )
		{
			glClear( GL_DEPTH_BUFFER_BIT );
		}
	}
	else
	{
		log_manager->log( LogManager::FBO_MANAGER, "Render Target \"%s\" not registered.", tname.c_str() );
	}
}
//
//============================================================================================================
//
void FboManager::setTargets( string	tname1,
							 string	tname2,
							 bool	clear_color_bit,
							 bool	clear_depth_bit
						   )
{
	if( targets_map.find( tname1 ) != targets_map.end() && targets_map.find( tname2 ) != targets_map.end() )
	{
		GLenum targets[] = { targets_map[tname1], targets_map[tname2] };
		glDrawBuffers( 2, targets );
		if( clear_color_bit )
		{
			glClear( GL_COLOR_BUFFER_BIT );
		}
		if( clear_depth_bit )
		{
			glClear( GL_DEPTH_BUFFER_BIT );
		}
	}
	else
	{
		log_manager->log( LogManager::FBO_MANAGER, "Render Target \"%s\" or \"%s\" not registered.",
												   tname1.c_str(),
												   tname2.c_str()										);
	}
}
//
//============================================================================================================
//
void FboManager::pushActiveBind( string			tex_name,
								 unsigned int	opengl_texture_offset
							   )
{
	pushed_tex_targets.push_back( tex_fbo_map[tex_name].fbo_tex_target );
	pushed_tex_offsets.push_back( opengl_texture_offset );

	glActiveTexture	( GL_TEXTURE0 + pushed_tex_offsets.back() );
	glBindTexture	( pushed_tex_targets.back(), texture_ids[tex_name] );
}
//
//============================================================================================================
//
void FboManager::popActiveBind( void )
{
	glActiveTexture	( GL_TEXTURE0 + pushed_tex_offsets.back() );
	glBindTexture	( pushed_tex_targets.back(), 0 );
	pushed_tex_targets.pop_back();
	pushed_tex_offsets.pop_back();
}
//
//============================================================================================================
//
bool FboManager::init( unsigned int display_width, unsigned int display_height, unsigned int agents_npot )
{
	for( unsigned int i = 0; i < inputs.size(); i++ )
	{
		LocalFbo local_fbo;
		FramebufferObject* fbo		= new FramebufferObject();
		local_fbo.fbo				= fbo;
		local_fbo.fbo_tex_target	= inputs[i]->fbo_tex_target;
		local_fbo.fbo_name			= inputs[i]->fbo_name;
		local_fbo.dynamic_size		= false;
		if( inputs[i]->display_width_flag )
		{
			inputs[i]->fbo_width	= (unsigned int)(display_width  * inputs[i]->wFactor);
			local_fbo.dynamic_size  = true;
		}
		if( inputs[i]->display_height_flag )
		{
			inputs[i]->fbo_height	= (unsigned int)(display_height * inputs[i]->hFactor);
			local_fbo.dynamic_size  = true;
		}
		if( inputs[i]->agents_npot_width_flag )
		{
			inputs[i]->fbo_width	= (unsigned int)(agents_npot    * inputs[i]->wFactor);
		}
		if( inputs[i]->agents_npot_height_flag )
		{
			inputs[i]->fbo_height	= (unsigned int)(agents_npot    * inputs[i]->hFactor);
		}
		local_fbo.fbo_width			= inputs[i]->fbo_width;
		local_fbo.fbo_height		= inputs[i]->fbo_height;
		fbos[inputs[i]->fbo_name]	= local_fbo;
#if defined(_WIN32)
		log_manager->log( LogManager::FBO_MANAGER, "%s::Width=%ipx, Height=%ipx.",
												   local_fbo.fbo_name.c_str(),
												   local_fbo.fbo_width,
												   local_fbo.fbo_height			);
#elif defined(__unix)
		log_manager->log( LogManager::FBO_MANAGER, "%s::Width=%ipx, Height=%ipx.",
												   local_fbo.fbo_name.c_str(),
												   local_fbo.fbo_width,
												   local_fbo.fbo_height			);
#endif
		local_fbo.fbo->Bind();
		{
			unsigned int curr_color_attachments = 0;
			for( unsigned int t = 0; t < inputs[i]->fbo_textures.size(); t++ )
			{
				GLuint tex_id = 0;
				texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name] = tex_id;
				tex_fbo_map[inputs[i]->fbo_textures[t]->fbo_texture_name] = local_fbo;
				string tex_msg = "While setting '";
				tex_msg.append( inputs[i]->fbo_textures[t]->fbo_texture_name );
				tex_msg.append( "'" );
				switch (inputs[i]->fbo_textures[t]->fbo_texture_type)
				{
					case InputFbo::TYPICAL:
						typicalTextureSettings( (char*)tex_msg.c_str(),
												local_fbo.fbo_tex_target,
												texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name],
												local_fbo.fbo_width,
												local_fbo.fbo_height										);
						if( maxAttachments > (int)curr_color_attachments )
						{
							registerColorTarget( inputs[i]->fbo_textures[t]->fbo_texture_name,
												 local_fbo.fbo_tex_target,
												 curr_color_attachments,
												 texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name] );
							curr_color_attachments++;
						}
						else
						{
							local_fbo.fbo->Disable();
							return false;
						}
						break;
					case InputFbo::GPGPU:
						gpgpuTextureSettings( (char*)tex_msg.c_str(),
											  local_fbo.fbo_tex_target,
											  texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name],
											  local_fbo.fbo_width,
											  local_fbo.fbo_height										);
						if( maxAttachments > (int)curr_color_attachments )
						{
							registerColorTarget( inputs[i]->fbo_textures[t]->fbo_texture_name,
												 local_fbo.fbo_tex_target,
												 curr_color_attachments,
												 texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name] );
							curr_color_attachments++;
						}
						else
						{
							local_fbo.fbo->Disable();
							return false;
						}
						break;
					case InputFbo::DEPTH_NO_COMPARE:
						depthTextureNoCompareSettings( (char*)tex_msg.c_str(),
													   local_fbo.fbo_tex_target,
													   texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name],
													   local_fbo.fbo_width,
													   local_fbo.fbo_height											);
						registerDepthTarget( local_fbo.fbo_tex_target,
											 texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name]				);
						break;
					case InputFbo::DEPTH_R_COMPARE:
						depthTextureRCompareSettings( (char*)tex_msg.c_str(),
													  local_fbo.fbo_tex_target,
													  texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name],
													  local_fbo.fbo_width,
													  local_fbo.fbo_height											);
						registerDepthTarget( local_fbo.fbo_tex_target,
											 texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name]				);
						break;
				}
			}
#if defined(_WIN32)
			log_manager->log( LogManager::FBO_MANAGER, "%s::%s.",
													   local_fbo.fbo_name.c_str(),
													   isFboValid( local_fbo.fbo->GetID() ) );
#elif defined(__unix)
			log_manager->log( LogManager::FBO_MANAGER, "%s::%s.",
													   local_fbo.fbo_name.c_str(),
													   isFboValid( local_fbo.fbo->GetID() ) );
#endif
			log_manager->separator();
		}
		local_fbo.fbo->Disable();
		if( !fboOK )
		{
			return false;
		}
		else
		{
			local_fbos.push_back( local_fbo );
		}
	}
	return true;
}
//
//============================================================================================================
//
bool FboManager::init( void )
{
	for( unsigned int i = 0; i < inputs.size(); i++ )
	{
		LocalFbo local_fbo;
		FramebufferObject* fbo    = new FramebufferObject();
		local_fbo.fbo			  = fbo;
		local_fbo.fbo_tex_target  = inputs[i]->fbo_tex_target;
		local_fbo.fbo_name		  = inputs[i]->fbo_name;
		local_fbo.fbo_width		  = inputs[i]->fbo_width;
		local_fbo.fbo_height	  = inputs[i]->fbo_height;
		fbos[inputs[i]->fbo_name] = local_fbo;
		log_manager->log( LogManager::FBO_MANAGER, "%s::Width=%ipx, Height=%ipx.",
												   local_fbo.fbo_name.c_str(),
												   local_fbo.fbo_width,
												   local_fbo.fbo_height			);
		local_fbo.fbo->Bind();
		{
			unsigned int curr_color_attachments = 0;
			for( unsigned int t = 0; t < inputs[i]->fbo_textures.size(); t++ )
			{
				GLuint tex_id = 0;
				texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name] = tex_id;
				tex_fbo_map[inputs[i]->fbo_textures[t]->fbo_texture_name] = local_fbo;
				switch (inputs[i]->fbo_textures[t]->fbo_texture_type)
				{
					char tex_msg[100];
					sprintf(	tex_msg,
								"While setting '%s'",
								inputs[i]->fbo_textures[t]->fbo_texture_name.c_str() );
					case InputFbo::TYPICAL:
						typicalTextureSettings( tex_msg,
												local_fbo.fbo_tex_target,
												texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name],
												local_fbo.fbo_width,
												local_fbo.fbo_height										);
						if( maxAttachments > (int)curr_color_attachments )
						{
							registerColorTarget( inputs[i]->fbo_textures[t]->fbo_texture_name,
												 local_fbo.fbo_tex_target,
												 curr_color_attachments,
												 texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name] );
							curr_color_attachments++;
						}
						else
						{
							local_fbo.fbo->Disable();
							return false;
						}
						break;
					case InputFbo::GPGPU:
						gpgpuTextureSettings( tex_msg,
											  local_fbo.fbo_tex_target,
											  texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name],
											  local_fbo.fbo_width,
											  local_fbo.fbo_height										);
						if( maxAttachments > (int)curr_color_attachments )
						{
							registerColorTarget( inputs[i]->fbo_textures[t]->fbo_texture_name,
												 local_fbo.fbo_tex_target,
												 curr_color_attachments,
												 texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name] );
							curr_color_attachments++;
						}
						else
						{
							local_fbo.fbo->Disable();
							return false;
						}
						break;
					case InputFbo::DEPTH_NO_COMPARE:
						depthTextureNoCompareSettings( tex_msg,
													   local_fbo.fbo_tex_target,
													   texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name],
													   local_fbo.fbo_width,
													   local_fbo.fbo_height											);
						registerDepthTarget( local_fbo.fbo_tex_target,
											 texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name]				);
						break;
					case InputFbo::DEPTH_R_COMPARE:
						depthTextureRCompareSettings( tex_msg,
													  local_fbo.fbo_tex_target,
													  texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name],
													  local_fbo.fbo_width,
													  local_fbo.fbo_height											);
						registerDepthTarget( local_fbo.fbo_tex_target,
											 texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name]				);
						break;
				}
			}
			local_fbo.fbo->Disable();
			log_manager->log( LogManager::FBO_MANAGER, "%s::%s.",
													   local_fbo.fbo_name.c_str(),
													   isFboValid( local_fbo.fbo->GetID() ) );
		}
		if( !fboOK )
		{
			return false;
		}
	}
	return true;
}
//
//============================================================================================================
//
bool FboManager::updateDisplayDims( unsigned int display_width, unsigned int display_height )
{
	bool result = false;
	for( unsigned int f = 0; f < local_fbos.size(); f++ )
	{
		if( local_fbos[f].dynamic_size )
		{
			result = updateFboDims( local_fbos[f].fbo_name, display_width, display_height );
			if( !result )
			{
				log_manager->log( LogManager::FBO_MANAGER, "Error while updating FBO: \"%s\".",
														   local_fbos[f].fbo_name.c_str()		);
				break;
			}
		}
	}
	if( result )
	{
#if defined(_WIN32)
		log_manager->log( LogManager::FBO_MANAGER, "Reshaped to %ix%ipx.", display_width, display_height );
#elif defined(__unix)
		log_manager->log( LogManager::FBO_MANAGER, "Reshaped to %ix%ipx.", display_width, display_height );
#endif
	}
	return result;
}
//
//============================================================================================================
//
bool FboManager::updateFboDims(	string			fbo_name,
								unsigned int	new_width,
								unsigned int	new_height	)
{
	for( unsigned int i = 0; i < inputs.size(); i++ )
	{
		if( fbo_name.compare( inputs[i]->fbo_name ) == 0 )
		{
			delete fbos[fbo_name].fbo;
			fbos[fbo_name].fbo		    = 0;
			if( inputs[i]->display_width_flag )
			{
				inputs[i]->fbo_width	= (unsigned int)(new_width  * inputs[i]->wFactor);
			}
			if( inputs[i]->display_height_flag )
			{
				inputs[i]->fbo_height	= (unsigned int)(new_height * inputs[i]->hFactor);
			}
			FramebufferObject* fbo		= new FramebufferObject();
			fbos[fbo_name].fbo			= fbo;
			fbos[fbo_name].fbo_width	= inputs[i]->fbo_width;
			fbos[fbo_name].fbo_height	= inputs[i]->fbo_height;
#if defined (_WIN32)
			log_manager->file_log( LogManager::FBO_MANAGER, "Updating \"%s\"::New_Width=%ipx, New_Height=%ipx.",
															fbos[fbo_name].fbo_name.c_str(),
															fbos[fbo_name].fbo_width,
															fbos[fbo_name].fbo_height							);
#elif (__unix)
			log_manager->file_log( LogManager::FBO_MANAGER, "Updating \"%s\"::New_Width=%ipx, New_Height=%ipx.",
																		fbos[fbo_name].fbo_name.c_str(),
																		fbos[fbo_name].fbo_width,
																		fbos[fbo_name].fbo_height				);
#endif
			fbos[fbo_name].fbo->Bind();
			{
				unsigned int curr_color_attachments = 0;
				for( unsigned int t = 0; t < inputs[i]->fbo_textures.size(); t++ )
				{
					FREE_TEXTURE( texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name] );
					texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name] = 0;
					string tex_msg = "While setting '";
					tex_msg.append( inputs[i]->fbo_textures[t]->fbo_texture_name );
					tex_msg.append( "'" );
					switch (inputs[i]->fbo_textures[t]->fbo_texture_type)
					{
						case InputFbo::TYPICAL:
							typicalTextureSettings( (char*)tex_msg.c_str(),
													fbos[fbo_name].fbo_tex_target,
													texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name],
													fbos[fbo_name].fbo_width,
													fbos[fbo_name].fbo_height									);
							if( maxAttachments > (int)curr_color_attachments )
							{
								registerColorTarget( inputs[i]->fbo_textures[t]->fbo_texture_name,
													 fbos[fbo_name].fbo_tex_target,
													 curr_color_attachments,
													 texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name]	);
								curr_color_attachments++;
							}
							else
							{
								fbos[fbo_name].fbo->Disable();
								return false;
							}
							break;
						case InputFbo::GPGPU:
							gpgpuTextureSettings( (char*)tex_msg.c_str(),
												  fbos[fbo_name].fbo_tex_target,
												  texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name],
												  fbos[fbo_name].fbo_width,
												  fbos[fbo_name].fbo_height										);
							if( maxAttachments > (int)curr_color_attachments )
							{
								registerColorTarget( inputs[i]->fbo_textures[t]->fbo_texture_name,
													 fbos[fbo_name].fbo_tex_target,
													 curr_color_attachments,
													 texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name]	);
								curr_color_attachments++;
							}
							else
							{
								fbos[fbo_name].fbo->Disable();
								return false;
							}
							break;
						case InputFbo::DEPTH_NO_COMPARE:
							depthTextureNoCompareSettings( (char*)tex_msg.c_str(),
														   fbos[fbo_name].fbo_tex_target,
														   texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name],
														   fbos[fbo_name].fbo_width,
														   fbos[fbo_name].fbo_height									);
							registerDepthTarget( fbos[fbo_name].fbo_tex_target,
												 texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name]				);
							break;
						case InputFbo::DEPTH_R_COMPARE:
							depthTextureRCompareSettings( (char*)tex_msg.c_str(),
														  fbos[fbo_name].fbo_tex_target,
														  texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name],
														  fbos[fbo_name].fbo_width,
														  fbos[fbo_name].fbo_height										);
							registerDepthTarget( fbos[fbo_name].fbo_tex_target,
												 texture_ids[inputs[i]->fbo_textures[t]->fbo_texture_name]				);
							break;
					}
				}
#if defined (_WIN32)
				log_manager->file_log( LogManager::FBO_MANAGER, "%s::%s.",
																fbos[fbo_name].fbo_name.c_str(),
																isFboValid( fbos[fbo_name].fbo->GetID() ) );
#elif defined (__unix)
				log_manager->file_log( LogManager::FBO_MANAGER, "%s::%s.",
																				fbos[fbo_name].fbo_name.c_str(),
																				isFboValid( fbos[fbo_name].fbo->GetID() ) );
#endif
			}
			fbos[fbo_name].fbo->Disable();
			if( !fboOK )
			{
				return false;
			}
			return true;
		}
	}
	log_manager->log( LogManager::FBO_MANAGER, "ERROR::WHILE_UPDATING::FBO \"%s\" not found.", fbo_name.c_str() );
	return false;
}
//
//============================================================================================================
//
void FboManager::bind_fbo( string fbo_name )
{
	fbos[fbo_name].fbo->Bind();
}
//
//============================================================================================================
//
void FboManager::unbind_fbo( string fbo_name )
{
	fbos[fbo_name].fbo->Disable();
}
//
//============================================================================================================
//
void FboManager::pushMatrices( Camera* light_cam )
{
	light_cam->readMatrices();
	glMatrixMode( GL_TEXTURE );
	glPushMatrix();
		glActiveTexture( GL_TEXTURE0 );
		glLoadIdentity();
		glLoadMatrixd( light_cam->projectionMatrix );
		glMultMatrixd( light_cam->modelviewMatrix );
	glMatrixMode( GL_MODELVIEW );
}
//
//============================================================================================================
//
void FboManager::popMatrices( void )
{
	glMatrixMode( GL_TEXTURE );
		glLoadIdentity();
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
}
//
//============================================================================================================
//
unsigned int FboManager::getWidth( string fbo_name )
{
	if( fbos.find( fbo_name ) != fbos.end() )
	{
		return fbos[fbo_name].fbo_width;
	}
	else
	{
		log_manager->log( LogManager::FBO_MANAGER, "FBO_WIDTH::FBO \"%s\" not registered in FboManager.",
												   fbo_name.c_str()										);
		return 0;
	}
}
//
//============================================================================================================
//
unsigned int FboManager::getHeight( string fbo_name )
{
	if( fbos.find( fbo_name ) != fbos.end() )
	{
		return fbos[fbo_name].fbo_height;
	}
	else
	{
		log_manager->log( LogManager::FBO_MANAGER, "FBO_HEIGHT::FBO \"%s\" not registered in FboManager.",
												   fbo_name.c_str()										);
		return 0;
	}
}
//
//============================================================================================================
//
void FboManager::report( void )
{
	map<string,LocalFbo>::iterator it;
	log_manager->file_log( LogManager::FBO_MANAGER, "FBO_MANAGER_REPORT:" );
	for( it = tex_fbo_map.begin(); it != tex_fbo_map.end(); it++ )
	{
#if defined(_WIN32)
		log_manager->file_log( LogManager::FBO_MANAGER, "FBO: '%s'. TEXTURE: '%s'. "
														"WIDTH: %upx. HEIGHT: %upx.",
														(*it).second.fbo_name.c_str(),
														(*it).first.c_str(),
														(*it).second.fbo_width,
														(*it).second.fbo_height		);
#elif defined(__unix)
		log_manager->file_log( LogManager::FBO_MANAGER, "FBO: '%s'. TEXTURE: '%s'. "
														"WIDTH: %upx. HEIGHT: %upx.",
														(*it).second.fbo_name.c_str(),
														(*it).first.c_str(),
														(*it).second.fbo_width,
														(*it).second.fbo_height		);
#endif
	}
}
//
//============================================================================================================
