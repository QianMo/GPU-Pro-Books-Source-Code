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


#include "cTextureManager.h"

//=======================================================================================

TextureManager* TextureManager::singleton = 0;

//=======================================================================================
//
TextureManager* TextureManager::getInstance( void )
{
	if( singleton == 0 )
	{
		singleton = new TextureManager;
	}
	return ((TextureManager *)singleton);
}
//
//=======================================================================================
//
void TextureManager::freeInstance( void )
{
	FREE_INSTANCE( singleton );
}
//
//=======================================================================================
//
void TextureManager::init( 	GlErrorManager*	err_manager_,
							LogManager*		log_manager_	)
{
	err_manager 	= err_manager_;
	log_manager		= log_manager_;
	curr_tex_width	= 64;
	curr_tex_height = 64;
	curr_tex_depth	= 1;
	curr_tex_weight = curr_tex_width * curr_tex_height * curr_tex_depth * 4;
	ilInit();
	ilEnable( IL_ORIGIN_SET );
	// This is the first texture loaded. If a texture
	// can't be loaded, we use this instead
	// initialize only once!
	if( texture_list.size() == 0 )
	{
	    string def_name = string( "default" );
		Texture* tex = new Texture( def_name,
									curr_tex_width,
									curr_tex_height,
									curr_tex_weight,
									4,
									GL_TEXTURE_2D,
									false,
									false,
									false           );
		// Create and initialize the texture
		glGenTextures( 1, &tex->id );
		glBindTexture( GL_TEXTURE_2D, tex->id );

		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,		GL_REPEAT );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,		GL_REPEAT );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,	GL_LINEAR );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,	GL_LINEAR );

		texture_list.push_back( tex );

		// Create a checker for the default texture
		int i, j, c;				// temporary variable
		unsigned char *checker;		// texture data
		checker = new unsigned char[ 64 * 64 * 4 ];
		for( i = 0; i < 64; i++ )
		{
			for( j = 0; j < 64; j++ )
			{
				c = ( !(i & 8) ^ !(j & 8)) * 255;
				checker[ (i * 256) + (j * 4) + 0 ] = (unsigned char)c;
				checker[ (i * 256) + (j * 4) + 1 ] = (unsigned char)c;
				checker[ (i * 256) + (j * 4) + 2 ] = (unsigned char)c;
				checker[ (i * 256) + (j * 4) + 3 ] = (unsigned char)255;
			}
		}
		glTexImage2D( GL_TEXTURE_2D,
					  0,
					  4,
					  64,
					  64,
					  0,
					  GL_RGBA,
					  GL_UNSIGNED_BYTE,
					  checker );
		delete [] checker;
	}
}
//
//=======================================================================================
//
void TextureManager::bindTexture( unsigned int texTarget, unsigned int id )
{
	glBindTexture( texTarget, id );
}
//
//=======================================================================================
//
unsigned int TextureManager::loadTexture( string&			filename,
										  bool				flip,
										  unsigned int		texTarget,
										  unsigned int		envMode		)
{
	string base_name = StringUtils::getStrNameFromPath( (char*)filename.c_str() );
	string env_str;
	GLenum env;
	switch( envMode )
	{
		case GL_MODULATE:
			env_str = string( "While setting ENV_MODE: GL_MODULATE" );
			env = GL_MODULATE;
			break;
		case GL_DECAL:
			env_str = string( "While setting ENV_MODE: GL_DECAL" );
			env = GL_DECAL;
			break;
		case GL_BLEND:
			env_str = string( "While setting ENV_MODE: GL_BLEND" );
			env = GL_BLEND;
			break;
		case GL_REPLACE:
			env_str = string( "While setting ENV_MODE: GL_REPLACE" );
			env = GL_REPLACE;
			break;
		default:
			log_manager->log( LogManager::LERROR, 	"Invalid ENV_MODE: (%#X) for \"%s\".",
													envMode,
													base_name.c_str()					);
			env_str = string( "While setting ENV_MODE: GL_MODULATE" );
			env = GL_MODULATE;
			break;
	}

	if( filename.compare( "default" ) != 0 )
	{
		log_manager->log( LogManager::TEXTURE_MANAGER, 	"Loading file: \"%s\" with ENV_MODE: %#X...",
														base_name.c_str(),
														env 										);
	}
	unsigned int	img_id	= 0;
	unsigned int	tex_id	= 0;
	int				bpp		= 0;
	int				width	= 0;
	int				height	= 0;
	int				success = 1;
	int				format	= 0;

	for( tListItor itor = texture_list.begin(); itor != texture_list.end(); ++itor )
	{
		if( (*itor)->getName().compare( filename ) == 0 )
		{
			if( filename.compare( "default" ) != 0 )
			{
				log_manager->log( LogManager::INFORMATION,	"Texture file: \"%s\" already loaded.",
															base_name.c_str() 						);
			}
			return (*itor)->getId();
		}
	}
	ilGenImages( 1, &img_id );
	ilBindImage( img_id );
	success = ilLoadImage( filename.c_str() );
	if( flip ) iluFlipImage();
	width	= ilGetInteger( IL_IMAGE_WIDTH );
	height	= ilGetInteger( IL_IMAGE_HEIGHT );
	format	= ilGetInteger( IL_IMAGE_FORMAT );
	bpp		= ilGetInteger( IL_IMAGE_BYTES_PER_PIXEL );
	if( success == IL_TRUE )
	{
		glGenTextures( 1, &tex_id );
		glBindTexture( texTarget, tex_id );
		// GL_MODULATE | GL_DECAL | GL_BLEND | GL_REPLACE
		glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, env );
		err_manager->getError( (char *)env_str.c_str() );
		if( texTarget == GL_TEXTURE_2D )
		{
			glTexParameteri( texTarget,
							 GL_TEXTURE_MIN_FILTER,
							 GL_LINEAR_MIPMAP_NEAREST );
		}
		else
		{
			glTexParameteri( texTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
		}
		glTexParameteri( texTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
		glTexParameteri( texTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
		glTexParameteri( texTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
		err_manager->getError( "While setting parameters:" );
		if( texTarget == GL_TEXTURE_2D )
		{
			if( GL_GENERATE_MIPMAP )
			{
				glTexParameteri( texTarget, GL_GENERATE_MIPMAP, GL_TRUE );
				glHint( GL_GENERATE_MIPMAP_HINT, GL_NICEST );
				glTexImage2D( texTarget,
							  0,
							  bpp,
							  width,
							  height,
							  0,
							  format,
							  GL_UNSIGNED_BYTE,
							  ilGetData() );
			}
			else
			{
				gluBuild2DMipmaps( texTarget,
								   bpp,
								   width,
								   height,
								   format,
								   GL_UNSIGNED_BYTE,
								   ilGetData() );
			}
			err_manager->getError( "While building mipmaps:" );
		}
		Texture *tex = new Texture( tex_id,
									filename,
									width,
									height,
									width * height * bpp,
									bpp,
									texTarget,
									flip,
									false,
									true );
		texture_list.push_back( tex );
	}
	else
	{
		// Can't load the texture, use default texture
		log_manager->log( LogManager::LERROR, "While loading texture file: \"%s\"!", filename.c_str() );
/*
		ILenum Error;
		while( (Error = ilGetError()) != IL_NO_ERROR )
		{
			log_manager->log( LogManager::LERROR, "IL(%d): %s!", Error, (char*)iluErrorString( Error ) );
		}*/
		tex_id = (*texture_list.begin())->getId();
	}
	if( success == IL_TRUE )
	{
		ilDeleteImages( 1, &img_id );
	}
	log_manager->log( LogManager::TEXTURE_MANAGER, 	"Done. Width: %i, Height: %i (%i MB) @%i BPP.",
													width,
													height,
													BYTE2MB(width * height * bpp),
													bpp );
	curr_tex_width	= width;
	curr_tex_height = height;
	curr_tex_depth	= 1;
	curr_tex_weight = curr_tex_width * curr_tex_height * curr_tex_depth * bpp;
	return tex_id;
}
//
//=======================================================================================
//
unsigned int TextureManager::loadTexture3D( vector<string>&	filenames,
										    vector<GLint>&	parameters,
											bool			flip,
											unsigned int	texTarget 	)
{
//->RETURN_IT_IF_ALREADY_EXISTS
	for( t3DListItor itor3D = texture3D_list.begin(); itor3D != texture3D_list.end(); ++itor3D )
	{
		bool all_match = true;
		vector<string> stored_filenames = (*itor3D)->getFileNames();
		if( stored_filenames.size() == filenames.size() )
		{
			for( unsigned int n = 0; n < filenames.size(); n++ )
			{
				if( filenames[n].compare( stored_filenames[n] ) != 0 )
				{
					all_match = false;
					break;
				}
			}
		}
		else
		{
			all_match = false;
		}
		if( all_match )
		{
			string msg = string( "" );
			msg.append( "Texture3D file: [" );
			for( unsigned int n = 0; n < filenames.size(); n++ )
			{
				string base_name = StringUtils::getStrNameFromPath( (char*)filenames[n].c_str() );
				msg.append( "\"" );
				msg.append( base_name );
				msg.append( "\"" );
				if( n+1 < filenames.size() )
				{
					msg.append( " | " );
				}
			}
			msg.append( "] already loaded." );
			log_manager->log( LogManager::INFORMATION, msg.c_str() );
			return (*itor3D)->getId();
		}
	}
//<-RETURN_IT_IF_ALREADY_EXISTS

	bool global_success			= true;
	unsigned int global_bpp		= 0;
	unsigned int global_width	= 0;
	unsigned int global_height	= 0;
	unsigned int global_depth	= filenames.size();
	unsigned int global_format	= 0;
	unsigned int global_tex_id	= 0;

	string base_name = StringUtils::getStrNameFromPath( (char*)filenames[0].c_str() );
	string env_str;
	log_manager->log( LogManager::TEXTURE_MANAGER, "Loading 3D texture \"%s\"...", (char*)base_name.c_str() );

	//err_manager->getError( "Before setting parameters:" );
	glGenTextures( 1, &global_tex_id );
	glBindTexture( texTarget, global_tex_id );
	for( unsigned int i = 0; i < filenames.size(); i++ )
	{
		unsigned int	img_id	= 0;
		int				width	= 0;
		int				height	= 0;
		int				format	= 0;
		int				bpp		= 0;
		int				success = 1;

		ilGenImages( 1, &img_id );
		ilBindImage( img_id );
		success = ilLoadImage( filenames[i].data() );
		if( success == IL_TRUE )
		{
			if( flip ) iluFlipImage();
			width	= ilGetInteger( IL_IMAGE_WIDTH );
			height	= ilGetInteger( IL_IMAGE_HEIGHT );
			format	= ilGetInteger( IL_IMAGE_FORMAT );
			bpp		= ilGetInteger( IL_IMAGE_BYTES_PER_PIXEL );
			if( i == 0 )
			{
				global_width = width;
				global_height = height;
				global_format = format;
				global_bpp = bpp;
			}
			else
			{
				if( width != (int)global_width )
				{
					global_success = false;
					log_manager->log( LogManager::LERROR, "\"%s\" differs in width!", filenames[i].c_str() );
					break;
				}
				if( height != (int)global_height )
				{
					global_success = false;
					log_manager->log( LogManager::LERROR, "\"%s\" differs in height!", filenames[i].c_str() );
					break;
				}
				if( format != (int)global_format )
				{
					global_success = false;
					log_manager->log( LogManager::LERROR, "\"%s\" differs in format!", filenames[i].c_str() );
					break;
				}
			}

			if( i == 0 )
			{
				// http://www.gamedev.net/topic/607286-glsl-texture-array/
				// Create a TEXTURE_2D_ARRAY, making sure the texture width and height are the dimensions of your largest texture.
				// The last argument is NULL as we want to iterate through our images and put them in the correct position in our texture array using glTexSubImage
				glTexImage3D( texTarget, 0, global_bpp, global_width, global_height, global_depth, 0, global_format, GL_UNSIGNED_BYTE, NULL );
				err_manager->getError( "AFTER: glTexImage3D" );
			}

			// Bind the texture (using GL_TEXTURE_2D_ARRAY as the texture type) and use glTexParameteri as usual
			// iterate through the images to put into your texture array
			glTexSubImage3D( texTarget, 0, 0, 0, i, global_width, global_height, 1, global_format, GL_UNSIGNED_BYTE, ilGetData() );
			err_manager->getError( "AFTER: glTexSubImage3D" );
			glTexParameteri( texTarget, GL_TEXTURE_MIN_FILTER, parameters[i] );
			glTexParameteri( texTarget, GL_TEXTURE_MAG_FILTER, parameters[i] );
			glTexParameteri( texTarget, GL_TEXTURE_WRAP_S, GL_REPEAT );
			glTexParameteri( texTarget, GL_TEXTURE_WRAP_T, GL_REPEAT );
			glTexParameteri( texTarget, GL_TEXTURE_WRAP_R, GL_REPEAT );
			err_manager->getError( "While setting parameters:" );
			ilDeleteImages( 1, &img_id );
		}
		else
		{
			global_success = false;
			log_manager->log( LogManager::LERROR, "While loading file \"%s\"!", (char*)filenames[i].data() );
			ILenum Error;
			while( (Error = ilGetError()) != IL_NO_ERROR )
			{
				log_manager->log( LogManager::LERROR, "IL(%d): %s!", Error, (char*)iluErrorString( Error ) );
			}
			break;
		}
	}

	if( global_success )
	{
		Texture3D *tex3D = new Texture3D( global_tex_id,
										  filenames[0],
										  filenames,
										  global_width,
										  global_height,
										  global_depth,
										  global_width * global_height * global_depth * global_bpp,
										  global_bpp,
										  texTarget,
										  flip,
										  false,
										  false														);
		texture3D_list.push_back( tex3D );
		log_manager->log( LogManager::TEXTURE_MANAGER, "Done. Width: %i, Height: %i, Depth: %i (%i MB) @%i BPP.",
														global_width,
														global_height,
														global_depth,
														BYTE2MB(global_width * global_height * global_depth * global_bpp),
														global_bpp														);
		curr_tex_width	= global_width;
		curr_tex_height = global_height;
		curr_tex_depth	= global_depth;
		curr_tex_weight = curr_tex_width * curr_tex_height * curr_tex_depth * global_bpp;
	}
	return global_tex_id;
}
//
//=======================================================================================
//
unsigned int TextureManager::loadRawTexture3D(	vector<string>&		filenames,
												vector<GLint>&		parameters,
												GLuint				width,
												GLint				internalFormat,	//GL_RGBA32F
												GLenum				format,			//GL_RGBA
												unsigned int		texTarget	)	//GL_TEXTURE_2D_ARRAY
{
//->RETURN_IT_IF_ALREADY_EXISTS
	for( t3DListItor itor3D = texture3D_list.begin(); itor3D != texture3D_list.end(); ++itor3D )
	{
		bool all_match = true;
		vector<string> stored_filenames = (*itor3D)->getFileNames();
		if( stored_filenames.size() == filenames.size() )
		{
			for( unsigned int n = 0; n < filenames.size(); n++ )
			{
				if( filenames[n].compare( stored_filenames[n] ) != 0 )
				{
					all_match = false;
					break;
				}
			}
		}
		else
		{
			all_match = false;
		}
		if( all_match )
		{
			string msg = string( "" );
			msg.append( "Texture3D file: [" );
			for( unsigned int n = 0; n < filenames.size(); n++ )
			{
				string base_name = StringUtils::getStrNameFromPath( (char*)filenames[n].c_str() );
				msg.append( "\"" );
				msg.append( base_name );
				msg.append( "\"" );
				if( n+1 < filenames.size() )
				{
					msg.append( " | " );
				}
			}
			msg.append( "] already loaded." );
			log_manager->log( LogManager::INFORMATION, msg.c_str() );
			return (*itor3D)->getId();
		}
	}
//<-RETURN_IT_IF_ALREADY_EXISTS

	bool global_success			= true;
	int global_bpp				= 8;
	unsigned int global_width	= width;
	unsigned int global_height	= 0;
	unsigned int global_depth	= filenames.size();
	GLenum global_format		= format;
	unsigned int global_tex_id	= 0;

	vector<string> sep = StringUtils::split( filenames[0], '/' );
	string base_name   = sep[sep.size() -1];

	log_manager->log( LogManager::TEXTURE_MANAGER, "Loading RAW 3D texture \"%s\"...", (char*)base_name.c_str() );

	string env_str;

	err_manager->getError( "Before setting parameters:" );
	glGenTextures( 1, &global_tex_id );
	glBindTexture( texTarget, global_tex_id );
	for( unsigned int i = 0; i < filenames.size(); i++ )
	{
		int	height	= 0;
		ifstream ifs( filenames[i].c_str(), std::ios::in | std::ios::binary );
		if( ifs.is_open() )
		{
			vector<float> data;
			float* line = new float[512];
			unsigned int channels = 4;
			unsigned int CHUNK = channels * width;

			while( !ifs.eof() )
			{
				ifs.read( (char*)line, (sizeof( float )*CHUNK) );
				if( !ifs.eof() )
				{
					for( unsigned int e = 0; e < CHUNK; e++ )
					{
						data.push_back( line[e] );
					}
					height++;
				}
			}
			delete [] line;
			ifs.close();

			if( i == 0 )
			{
				global_height = height;
				// http://www.gamedev.net/topic/607286-glsl-texture-array/
				// Create a TEXTURE_2D_ARRAY, making sure the texture width and height are the dimensions of your largest texture.
				// The last argument is NULL as we want to iterate through our images and put them in the correct position in our texture array using glTexSubImage
				glTexImage3D( texTarget, 0, internalFormat, global_width, global_height, global_depth, 0, global_format, GL_FLOAT, NULL );
				err_manager->getError( "AFTER: glTexImage3D" );
			}
			else
			{
				if( height != (int)global_height )
				{
					global_success = false;
					log_manager->log( LogManager::LERROR, "\"%s\" differs in height!", (char*)filenames[i].c_str() );
					break;
				}
			}

			// Bind the texture (using GL_TEXTURE_2D_ARRAY as the texture type) and use glTexParameteri as usual
			// iterate through the images to put into your texture array
			glTexSubImage3D( texTarget, 0, 0, 0, i, global_width, global_height, 1, global_format, GL_FLOAT, &data[0] );
			err_manager->getError( "AFTER: glTexSubImage3D" );
			glTexParameteri( texTarget, GL_TEXTURE_MIN_FILTER, parameters[i] );
			glTexParameteri( texTarget, GL_TEXTURE_MAG_FILTER, parameters[i] );
			glTexParameteri( texTarget, GL_TEXTURE_WRAP_S, GL_REPEAT );
			glTexParameteri( texTarget, GL_TEXTURE_WRAP_T, GL_REPEAT );
			glTexParameteri( texTarget, GL_TEXTURE_WRAP_R, GL_REPEAT );
			err_manager->getError( "While setting parameters:" );
		}
		else
		{
			global_success = false;
			log_manager->log( LogManager::LERROR, "While opening \"%s\"!", (char*)filenames[i].c_str() );
			break;
		}
	}

	if( global_success )
	{
		Texture3D *tex3D = new Texture3D( global_tex_id,
										  filenames[0],
										  filenames,
										  global_width,
										  global_height,
										  global_depth,
										  global_width * global_height * global_depth * global_bpp,
										  global_bpp,
										  texTarget,
										  false,
										  false,
										  false														);
		texture3D_list.push_back( tex3D );
		log_manager->log( LogManager::TEXTURE_MANAGER, 	"Done. Width: %i, Height: %i, Depth: %i (%i MB) @%i BPP.",
														global_width,
														global_height,
														global_depth,
														BYTE2MB(global_width * global_height * global_depth * global_bpp),
														global_bpp														);
		curr_tex_width	= global_width;
		curr_tex_height = global_height;
		curr_tex_depth	= global_depth;
		curr_tex_weight = curr_tex_width * curr_tex_height * curr_tex_depth * global_bpp;
	}
	return global_tex_id;
}
//
//=======================================================================================
//
unsigned int TextureManager::loadRawTexture(	string&			filename,
												GLint			parameter,
												GLuint			width,
												GLint			internalFormat,
												GLenum			format,
												unsigned int	texTarget	)
{
	string base_name = StringUtils::getStrNameFromPath( (char*)filename.c_str() );
	string env_str;

	log_manager->log( LogManager::TEXTURE_MANAGER, "Loading RAW texture \"%s\"...", (char*)base_name.c_str() );

	for( tListItor itor = texture_list.begin(); itor != texture_list.end(); ++itor )
	{
		if( (*itor)->getName().compare( filename ) == 0 )
		{
			if( filename.compare( "default" ) != 0 )
			{
				log_manager->log( LogManager::INFORMATION, "Texture file: \"%s\" already loaded.", (char*)base_name.c_str() );
			}
			return (*itor)->getId();
		}
	}

	int bpp				= 8;
	unsigned int height	= 0;
	unsigned int tex_id	= 0;

	err_manager->getError( "Before setting parameters:" );
	glGenTextures( 1, &tex_id );
	glBindTexture( texTarget, tex_id );

	ifstream ifs( filename.c_str(), std::ios::in | std::ios::binary );
	vector<float> data;
	unsigned int channels = 4;
	unsigned int CHUNK = channels * width;
	float* line = new float[512];
	while( !ifs.eof() )
	{
		ifs.read( (char*)line, (sizeof( float )*CHUNK) );
		if( !ifs.eof() )
		{
			for( unsigned int e = 0; e < CHUNK; e++ )
			{
				data.push_back( line[e] );
				line[e] = 0.0f;
			}
			height++;
		}
	}
	delete [] line;
	ifs.close();

	glGenTextures( 1, &tex_id );
	glBindTexture( texTarget, tex_id );
	glTexParameteri( texTarget, GL_TEXTURE_MIN_FILTER, parameter );
	glTexParameteri( texTarget, GL_TEXTURE_MAG_FILTER, parameter );
	glTexParameteri( texTarget, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri( texTarget, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D( texTarget, 0, internalFormat, width, height, 0, format, GL_FLOAT, &data[0] );
	glBindTexture( texTarget, 0 );
	err_manager->getError( "While setting parameters:" );

	Texture *tex = new Texture( tex_id,
								filename,
								width,
								height,
								width * height * bpp,
								bpp,
								texTarget,
								false,
								false,
								false				);
	texture_list.push_back( tex );
	log_manager->log( LogManager::TEXTURE_MANAGER, 	"Done. Width: %i, Height: %i (%i MB) @%i BPP.",
													width,
													height,
													BYTE2MB(width * height * bpp),
													bpp												);
	curr_tex_width	= width;
	curr_tex_height = height;
	curr_tex_depth	= 1;
	curr_tex_weight = curr_tex_width * curr_tex_height * bpp;
	return tex_id;
}
//
//=======================================================================================
//
unsigned int TextureManager::loadRawTexture(	string&			filename,
												vector<float>&	data,
												GLint			parameter,
												GLuint			width,
												GLint			internalFormat,
												GLenum			format,
												unsigned int	texTarget,
												bool			do_log			)
{
	string base_name = StringUtils::getStrNameFromPath( (char*)filename.c_str() );
	string env_str;

	if( do_log )
		log_manager->log( LogManager::TEXTURE_MANAGER, "Loading RAW texture \"%s\"...", (char*)base_name.c_str() );

	for( tListItor itor = texture_list.begin(); itor != texture_list.end(); ++itor )
	{
		if( (*itor)->getName().compare( filename ) == 0 )
		{
			if( filename.compare( "default" ) != 0 )
			{
				if( do_log )
					log_manager->log( LogManager::INFORMATION, "Texture file: \"%s\" already loaded.", (char*)base_name.c_str() );
			}
			return (*itor)->getId();
		}
	}

	int bpp				= 8;
	unsigned int height	= (data.size() / 4) / width;
	unsigned int tex_id	= 0;

	err_manager->getError( "Before setting parameters:" );
	glGenTextures( 1, &tex_id );
	glBindTexture( texTarget, tex_id );

	glGenTextures( 1, &tex_id );
	glBindTexture( texTarget, tex_id );
	glTexParameteri( texTarget, GL_TEXTURE_MIN_FILTER, parameter );
	glTexParameteri( texTarget, GL_TEXTURE_MAG_FILTER, parameter );
	glTexParameteri( texTarget, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri( texTarget, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D( texTarget, 0, internalFormat, width, height, 0, format, GL_FLOAT, &data[0] );
	glBindTexture( texTarget, 0 );
	err_manager->getError( "While setting parameters:" );

	Texture *tex = new Texture( tex_id,
								filename,
								width,
								height,
								width * height * bpp,
								bpp,
								texTarget,
								false,
								false,
								false				);
	texture_list.push_back( tex );
	if( do_log )
		log_manager->log( LogManager::TEXTURE_MANAGER, "Done. Width: %i, Height: %i (%i MB) @%i BPP.",
														width,
														height,
														BYTE2MB(width * height * bpp),
														bpp												);
	curr_tex_width	= width;
	curr_tex_height = height;
	curr_tex_depth	= 1;
	curr_tex_weight = curr_tex_width * curr_tex_height * bpp;
	return tex_id;
}
//
//=======================================================================================
//
unsigned int TextureManager::reloadRawTexture(	string&			filename,
												vector<float>&	data,
												GLint			parameter,
												GLuint			width,
												GLint			internalFormat,
												GLenum			format,
												unsigned int	texTarget	)
{
	deleteTexture( getTexture( filename ) );
	return loadRawTexture(	filename,
							data,
							parameter,
							width,
							internalFormat,
							format,
							texTarget,
							false			);
}
//
//=======================================================================================
//
unsigned int TextureManager::loadTexture( string&			filename,
										  bool				flip,
										  bool				mipmaps,
										  unsigned int		texTarget,
										  unsigned int		envMode,
										  unsigned int		filterMode
										)
{
	vector<string> sep = StringUtils::split( filename, '/' );
	string base_name   = sep[sep.size() -1];
	string env_str;
	string filter_str;
	GLenum env;
	GLenum filter;
	switch( envMode )
	{
		case GL_MODULATE:
			env_str = string( "While setting ENV_MODE: GL_MODULATE" );
			env = GL_MODULATE;
			break;
		case GL_DECAL:
			env_str = string( "While setting ENV_MODE: GL_DECAL" );
			env = GL_DECAL;
			break;
		case GL_BLEND:
			env_str = string( "While setting ENV_MODE: GL_BLEND" );
			env = GL_BLEND;
			break;
		case GL_REPLACE:
			env_str = string( "While setting ENV_MODE: GL_REPLACE" );
			env = GL_REPLACE;
			break;
		default:
			log_manager->log( LogManager::LERROR, 	"Invalid ENV_MODE: (%#X) for \"%s\".",
													envMode,
													(char*)base_name.c_str()			);
			env_str = string( "While setting ENV_MODE: GL_MODULATE" );
			env = GL_MODULATE;
			break;
	}

	switch ( filterMode )
	{
		case GL_NEAREST:
			filter_str = string( "While setting FILTER_MODE: GL_NEAREST" );
			filter = GL_NEAREST;
			break;
		case GL_LINEAR:
			filter_str = string( "While setting FILTER_MODE: GL_LINEAR" );
			filter = GL_LINEAR;
			break;
		default:
			log_manager->log( LogManager::LERROR, 	"Invalid FILTER_MODE: (%#X) for \"%s\".",
													filterMode,
													base_name.c_str()						);
			filter_str = string( "While setting FILTER_MODE: GL_LINEAR" );
			filter = GL_LINEAR;
			break;
	}

	if( filename.compare( "default" ) != 0 )
	{
		log_manager->log( LogManager::TEXTURE_MANAGER,	"Loading file: \"%s\" with ENV_MODE: %#X... ",
														base_name.c_str(),
														env 										);
	}
	unsigned int	img_id	= 0;
	unsigned int	tex_id	= 0;
	int				bpp		= 0;
	int				width	= 0;
	int				height	= 0;
	int				success = 1;
	int				format	= 0;

	for( tListItor itor = texture_list.begin(); itor != texture_list.end(); ++itor )
	{
		if( (*itor)->getName().compare( filename ) == 0 )
		{
			if( filename.compare( "default" ) != 0 )
			{
				log_manager->log( LogManager::INFORMATION, "Texture file: \"%s\" already loaded.", (char*)base_name.c_str() );
			}
			return (*itor)->getId();
		}
	}
	ilGenImages( 1, &img_id );
	ilBindImage( img_id );
	success = ilLoadImage( filename.c_str() );
	if( flip ) iluFlipImage();
	width	= ilGetInteger( IL_IMAGE_WIDTH );
	height	= ilGetInteger( IL_IMAGE_HEIGHT );
	format	= ilGetInteger( IL_IMAGE_FORMAT );
	bpp		= ilGetInteger( IL_IMAGE_BYTES_PER_PIXEL );
	if( success == IL_TRUE )
	{
		glGenTextures( 1, &tex_id );
		glBindTexture( texTarget, tex_id );
		// GL_MODULATE | GL_DECAL | GL_BLEND | GL_REPLACE
		glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, env );
		err_manager->getError( (char *)env_str.c_str() );
		if( texTarget == GL_TEXTURE_2D && mipmaps )
		{
			glTexParameteri( texTarget,
							 GL_TEXTURE_MIN_FILTER,
							 GL_LINEAR_MIPMAP_NEAREST );
		}
		else
		{
			glTexParameteri( texTarget, GL_TEXTURE_MIN_FILTER, filter );
			err_manager->getError( (char *)filter_str.c_str() );
		}
		glTexParameteri( texTarget, GL_TEXTURE_MAG_FILTER, filter );
		err_manager->getError( (char *)filter_str.c_str() );
		glTexParameteri( texTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
		glTexParameteri( texTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
		err_manager->getError( "While setting parameters:" );
		if( texTarget == GL_TEXTURE_2D && mipmaps)
		{
			if( GL_GENERATE_MIPMAP )
			{
				glTexParameteri( texTarget, GL_GENERATE_MIPMAP, GL_TRUE );
				glHint( GL_GENERATE_MIPMAP_HINT, GL_NICEST );
				glTexImage2D( texTarget,
							  0,
							  bpp,
							  width,
							  height,
							  0,
							  format,
							  GL_UNSIGNED_BYTE,
							  ilGetData() );
			}
			else
			{
				gluBuild2DMipmaps( texTarget,
								   bpp,
								   width,
								   height,
								   format,
								   GL_UNSIGNED_BYTE,
								   ilGetData() );
			}
			err_manager->getError( "While building mipmaps:" );
		}
		else
		{
			glTexImage2D( texTarget,
						  0,
						  bpp,
						  width,
						  height,
						  0,
						  format,
						  GL_UNSIGNED_BYTE,
						  ilGetData() );
		}
		Texture *tex = new Texture( tex_id,
									filename,
									width,
									height,
									width * height * bpp,
									bpp,
									texTarget,
									flip,
									false,
									true );
		texture_list.push_back( tex );
	}
	else
	{
		// Can't load the texture, use default texture
		log_manager->log( LogManager::LERROR, "While loading texture file: \"%s\"!", (char*)base_name.c_str() );
		tex_id = (*texture_list.begin())->getId();
	}
	if( success == IL_TRUE )
	{
		ilDeleteImages( 1, &img_id );
	}
	log_manager->log( LogManager::TEXTURE_MANAGER, 	"Done. Width: %i, Height: %i (%i MB) @%i BPP.",
													width,
													height,
													BYTE2MB(width * height * bpp),
													bpp											);
	curr_tex_width	= width;
	curr_tex_height = height;
	curr_tex_depth	= 1;
	curr_tex_weight = curr_tex_width * curr_tex_height * curr_tex_depth * bpp;
	return tex_id;
}
//
//=======================================================================================
//
unsigned int TextureManager::getTexture( string& filename )
{
	for( tListItor itor = texture_list.begin(); itor != texture_list.end(); ++itor )
	{
		if( (*itor)->getName().compare( filename ) == 0 )
		{
			return (*itor)->getId();
		}
	}
	string base_name = StringUtils::getStrNameFromPath( (char*)filename.c_str() );
	log_manager->log( LogManager::LERROR, 	"Texture ID not found for file: \"%s\"!",
											(char*)base_name.c_str() 				);
	return 0;
}
//
//=======================================================================================
//
unsigned int TextureManager::getTexture3D( string& filename )
{
	for( t3DListItor itor3D = texture3D_list.begin(); itor3D != texture3D_list.end(); ++itor3D )
	{
		if( (*itor3D)->getName().compare( filename ) == 0 )
		{
			return (*itor3D)->getId();
		}
	}
	string base_name = StringUtils::getStrNameFromPath( (char*)filename.c_str() );
	log_manager->log( LogManager::LERROR, 	"Texture ID not found for file: \"%s\"!",
											(char*)base_name.c_str() 				);
	return 0;
}
//
//=======================================================================================
//
textureList TextureManager::getTextures( void )
{
	return texture_list;
}
//
//=======================================================================================
//
texture3DList TextureManager::getTextures3D( void )
{
	return texture3D_list;
}
//
//=======================================================================================
//
unsigned int TextureManager::getCurrWidth()
{
	return curr_tex_width;
}
//
//=======================================================================================
//
unsigned int TextureManager::getCurrHeight()
{
	return curr_tex_height;
}
//
//=======================================================================================
//
unsigned int TextureManager::getCurrDepth()
{
	return curr_tex_depth;
}
//
//=======================================================================================
//
unsigned int TextureManager::getCurrWeight()
{
	return curr_tex_weight;
}
//
//=======================================================================================
//
void TextureManager::deleteTexture( unsigned int id )
{
	for( tListItor itor = texture_list.begin(); itor != texture_list.end(); ++itor )
	{
		if( (*itor)->getId() == id )
		{
			delete (*itor);
			itor = texture_list.erase( itor );
			break;
		}
	}
}
//
//=======================================================================================
//
void TextureManager::deleteTexture3D( unsigned int id )
{
	for( t3DListItor itor3D = texture3D_list.begin(); itor3D != texture3D_list.end(); ++itor3D )
	{
		if( (*itor3D)->getId() == id )
		{
			delete (*itor3D);
			itor3D = texture3D_list.erase( itor3D );
			break;
		}
	}
}
//
//=======================================================================================
//
void TextureManager::cleanAllTextures( void )
{
	tListItor itor = texture_list.begin();
	while( itor != texture_list.end() )
	{
		delete (*itor);
		itor = texture_list.erase( itor );
	}

	t3DListItor itor3D = texture3D_list.begin();
	while( itor3D != texture3D_list.end() )
	{
		delete (*itor3D);
		itor3D = texture3D_list.erase( itor3D );
	}
}
//
//=======================================================================================
