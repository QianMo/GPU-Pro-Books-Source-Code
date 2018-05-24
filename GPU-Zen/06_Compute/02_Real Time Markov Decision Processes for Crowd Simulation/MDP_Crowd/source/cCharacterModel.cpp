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
#include "cCharacterModel.h"

//=======================================================================================
//
CharacterModel::CharacterModel( void )
{
	name			= "";
	vbo_manager		= NULL;
	glsl_manager	= NULL;
	head			= NULL;
	hair			= NULL;
	torso			= NULL;
	legs			= NULL;
	model			= NULL;
	LOD				= LOD_HI;
}
//
//=======================================================================================
//
CharacterModel::CharacterModel( LOD_TYPE		_LOD,
								string			_name,
								VboManager*		_vbo_manager,
								GlslManager*	_glsl_manager,
								Model3D*		_head,
								Model3D*		_hair,
								Model3D*		_torso,
								Model3D*		_legs			)
{
	LOD							= _LOD;
	name						= string( _name );
	vbo_manager					= _vbo_manager;
	glsl_manager				= _glsl_manager;
	head						= _head;
	hair						= _hair;
	torso						= _torso;
	legs						= _legs;
	model						= NULL;
}
//
//=======================================================================================
//
CharacterModel::CharacterModel( LOD_TYPE		_LOD,
								string			_name,
								VboManager*		_vbo_manager,
								GlslManager*	_glsl_manager,
								Model3D*		_model			)
{
	LOD							= _LOD;
	name						= string( _name );
	vbo_manager					= _vbo_manager;
	glsl_manager				= _glsl_manager;
	model						= _model;
	head						= NULL;
	hair						= NULL;
	torso						= NULL;
	legs						= NULL;
}
//
//=======================================================================================
//
CharacterModel::~CharacterModel( void )
{

}
//
//=======================================================================================
//
bool CharacterModel::stitch_parts( void )
{
	if( LOD == LOD_HI || LOD == LOD_ME )
	{
		vector<Location4>	head_locations			= head->getUniqueLocations();
		vector<Normal>		head_normals			= head->getUniqueNormals();
		vector<Uv>			head_uvs				= head->getUniqueUvs();
		vector<Face3>		head_faces				= head->getFaces();

		vector<Location4>	hair_locations;
		vector<Normal>		hair_normals;
		vector<Uv>			hair_uvs;
		vector<Face3>		hair_faces;
		if( hair )
		{
			hair_locations							= hair->getUniqueLocations();
			hair_normals							= hair->getUniqueNormals();
			hair_uvs								= hair->getUniqueUvs();
			hair_faces								= hair->getFaces();
		}

		vector<Location4>	torso_locations			= torso->getUniqueLocations();
		vector<Normal>		torso_normals			= torso->getUniqueNormals();
		vector<Uv>			torso_uvs				= torso->getUniqueUvs();
		vector<Face3>		torso_faces				= torso->getFaces();

		vector<Location4>	legs_locations			= legs->getUniqueLocations();
		vector<Normal>		legs_normals			= legs->getUniqueNormals();
		vector<Uv>			legs_uvs				= legs->getUniqueUvs();
		vector<Face3>		legs_faces				= legs->getFaces();

		unsigned int		head_locations_offset	= 0;
		unsigned int		head_normals_offset		= 0;
		unsigned int		head_uvs_offset			= 0;

		unsigned int		hair_locations_offset	= head_locations.size();
		unsigned int		hair_normals_offset		= head_normals.size();
		unsigned int		hair_uvs_offset			= head_uvs.size();

		unsigned int		torso_locations_offset	= head_locations.size() + hair_locations.size();
		unsigned int		torso_normals_offset	= head_normals.size()	+ hair_normals.size();
		unsigned int		torso_uvs_offset		= head_uvs.size()		+ hair_uvs.size();

		unsigned int		legs_locations_offset	= head_locations.size() + hair_locations.size()	+ torso_locations.size();
		unsigned int		legs_normals_offset		= head_normals.size()	+ hair_normals.size()	+ torso_normals.size();
		unsigned int		legs_uvs_offset			= head_uvs.size()		+ hair_uvs.size()		+ torso_uvs.size();

	//->LOCATIONS
		for( unsigned int hl = 0; hl < head_locations.size(); hl++ )
		{
			unique_locations.push_back( head_locations[hl] );
		}
		for( unsigned int hl = 0; hl < hair_locations.size(); hl++ )
		{
			unique_locations.push_back( hair_locations[hl] );
		}
		for( unsigned int tl = 0; tl < torso_locations.size(); tl++ )
		{
			unique_locations.push_back( torso_locations[tl] );
		}
		for( unsigned int ll = 0; ll < legs_locations.size(); ll++ )
		{
			unique_locations.push_back( legs_locations[ll] );
		}
	//<-LOCATIONS

	//->NORMALS
		for( unsigned int hn = 0; hn < head_normals.size(); hn++ )
		{
			unique_normals.push_back( head_normals[hn] );
		}
		for( unsigned int hn = 0; hn < hair_normals.size(); hn++ )
		{
			unique_normals.push_back( hair_normals[hn] );
		}
		for( unsigned int tn = 0; tn < torso_normals.size(); tn++ )
		{
			unique_normals.push_back( torso_normals[tn] );
		}
		for( unsigned int ln = 0; ln < legs_normals.size(); ln++ )
		{
			unique_normals.push_back( legs_normals[ln] );
		}
	//<-NORMALS

	//->UVS
		for( unsigned int hu = 0; hu < head_uvs.size(); hu++ )
		{
			unique_uvs.push_back( head_uvs[hu] );
		}
		for( unsigned int hu = 0; hu < hair_uvs.size(); hu++ )
		{
			unique_uvs.push_back( hair_uvs[hu] );
		}
		for( unsigned int tu = 0; tu < torso_uvs.size(); tu++ )
		{
			unique_uvs.push_back( torso_uvs[tu] );
		}
		for( unsigned int lu = 0; lu < legs_uvs.size(); lu++ )
		{
			unique_uvs.push_back( legs_uvs[lu] );
		}
	//<-UVS

	//->FACES
		for( unsigned int hf = 0; hf < head_faces.size(); hf++ )
		{
			Face3 face3;
			INITFACE3( face3 );
			face3.location_indices[0]	= head_faces[hf].location_indices[0]	+ head_locations_offset;
			face3.location_indices[1]	= head_faces[hf].location_indices[1]	+ head_locations_offset;
			face3.location_indices[2]	= head_faces[hf].location_indices[2]	+ head_locations_offset;

			face3.normal_indices[0]		= head_faces[hf].normal_indices[0]		+ head_normals_offset;
			face3.normal_indices[1]		= head_faces[hf].normal_indices[1]		+ head_normals_offset;
			face3.normal_indices[2]		= head_faces[hf].normal_indices[2]		+ head_normals_offset;

			face3.uv_indices[0]			= head_faces[hf].uv_indices[0]			+ head_uvs_offset;
			face3.uv_indices[1]			= head_faces[hf].uv_indices[1]			+ head_uvs_offset;
			face3.uv_indices[2]			= head_faces[hf].uv_indices[2]			+ head_uvs_offset;

			faces.push_back( face3 );
		}

		for( unsigned int hf = 0; hf < hair_faces.size(); hf++ )
		{
			Face3 face3;
			INITFACE3( face3 );
			face3.location_indices[0]	= hair_faces[hf].location_indices[0]	+ hair_locations_offset;
			face3.location_indices[1]	= hair_faces[hf].location_indices[1]	+ hair_locations_offset;
			face3.location_indices[2]	= hair_faces[hf].location_indices[2]	+ hair_locations_offset;

			face3.normal_indices[0]		= hair_faces[hf].normal_indices[0]		+ hair_normals_offset;
			face3.normal_indices[1]		= hair_faces[hf].normal_indices[1]		+ hair_normals_offset;
			face3.normal_indices[2]		= hair_faces[hf].normal_indices[2]		+ hair_normals_offset;

			face3.uv_indices[0]			= hair_faces[hf].uv_indices[0]			+ hair_uvs_offset;
			face3.uv_indices[1]			= hair_faces[hf].uv_indices[1]			+ hair_uvs_offset;
			face3.uv_indices[2]			= hair_faces[hf].uv_indices[2]			+ hair_uvs_offset;

			faces.push_back( face3 );
		}

		for( unsigned int tf = 0; tf < torso_faces.size(); tf++ )
		{
			Face3 face3;
			INITFACE3( face3 );
			face3.location_indices[0]	= torso_faces[tf].location_indices[0]	+ torso_locations_offset;
			face3.location_indices[1]	= torso_faces[tf].location_indices[1]	+ torso_locations_offset;
			face3.location_indices[2]	= torso_faces[tf].location_indices[2]	+ torso_locations_offset;

			face3.normal_indices[0]		= torso_faces[tf].normal_indices[0]		+ torso_normals_offset;
			face3.normal_indices[1]		= torso_faces[tf].normal_indices[1]		+ torso_normals_offset;
			face3.normal_indices[2]		= torso_faces[tf].normal_indices[2]		+ torso_normals_offset;

			face3.uv_indices[0]			= torso_faces[tf].uv_indices[0]			+ torso_uvs_offset;
			face3.uv_indices[1]			= torso_faces[tf].uv_indices[1]			+ torso_uvs_offset;
			face3.uv_indices[2]			= torso_faces[tf].uv_indices[2]			+ torso_uvs_offset;

			faces.push_back( face3 );
		}

		for( unsigned int lf = 0; lf < legs_faces.size(); lf++ )
		{
			Face3 face3;
			INITFACE3( face3 );
			face3.location_indices[0]	= legs_faces[lf].location_indices[0]	+ legs_locations_offset;
			face3.location_indices[1]	= legs_faces[lf].location_indices[1]	+ legs_locations_offset;
			face3.location_indices[2]	= legs_faces[lf].location_indices[2]	+ legs_locations_offset;

			face3.normal_indices[0]		= legs_faces[lf].normal_indices[0]		+ legs_normals_offset;
			face3.normal_indices[1]		= legs_faces[lf].normal_indices[1]		+ legs_normals_offset;
			face3.normal_indices[2]		= legs_faces[lf].normal_indices[2]		+ legs_normals_offset;

			face3.uv_indices[0]			= legs_faces[lf].uv_indices[0]			+ legs_uvs_offset;
			face3.uv_indices[1]			= legs_faces[lf].uv_indices[1]			+ legs_uvs_offset;
			face3.uv_indices[2]			= legs_faces[lf].uv_indices[2]			+ legs_uvs_offset;

			faces.push_back( face3 );
		}
	//<-FACES
	}
	else if( LOD == LOD_LO )
	{
		vector<Location4>	model_locations		= model->getUniqueLocations();
		vector<Normal>		model_normals		= model->getUniqueNormals();
		vector<Uv>			model_uvs			= model->getUniqueUvs();
		vector<Face3>		model_faces			= model->getFaces();
		for( unsigned int ml = 0; ml < model_locations.size(); ml++ )
		{
			unique_locations.push_back( model_locations[ml] );
		}
		for( unsigned int mn = 0; mn < model_normals.size(); mn++ )
		{
			unique_normals.push_back( model_normals[mn] );
		}
		for( unsigned int mu = 0; mu < model_uvs.size(); mu++ )
		{
			unique_uvs.push_back( model_uvs[mu] );
		}
		for( unsigned int mf = 0; mf < model_faces.size(); mf++ )
		{
			Face3 face3;
			INITFACE3( face3 );
			face3.location_indices[0]	= model_faces[mf].location_indices[0];
			face3.location_indices[1]	= model_faces[mf].location_indices[1];
			face3.location_indices[2]	= model_faces[mf].location_indices[2];

			face3.normal_indices[0]		= model_faces[mf].normal_indices[0];
			face3.normal_indices[1]		= model_faces[mf].normal_indices[1];
			face3.normal_indices[2]		= model_faces[mf].normal_indices[2];

			face3.uv_indices[0]			= model_faces[mf].uv_indices[0];
			face3.uv_indices[1]			= model_faces[mf].uv_indices[1];
			face3.uv_indices[2]			= model_faces[mf].uv_indices[2];

			faces.push_back( face3 );
		}
	}

	// Prepare a VBO to fill:
	unsigned int vbo_frame = 0;
	string vbo_name = name;
	vbo_name.append( "_LOD" );
	string lod_str = static_cast<ostringstream*>( &(ostringstream() << LOD) )->str();
	vbo_name.append( lod_str );
	unsigned int vbo_index = vbo_manager->createVBOContainer( vbo_name, vbo_frame );

	for( unsigned int f = 0; f < faces.size(); f++ )
	{
		Face3 face3 = faces[f];

		Vertex v1;
		INITVERTEX( v1 );
		v1.location[0] = unique_locations[face3.location_indices[0]].location[0];
		v1.location[1] = unique_locations[face3.location_indices[0]].location[1];
		v1.location[2] = unique_locations[face3.location_indices[0]].location[2];
		v1.normal[0] = unique_normals[face3.normal_indices[0]].normal[0];
		v1.normal[1] = unique_normals[face3.normal_indices[0]].normal[1];
		v1.normal[2] = unique_normals[face3.normal_indices[0]].normal[2];
		v1.texture[0] = unique_uvs[face3.uv_indices[0]].uv[0];
		v1.texture[1] = unique_uvs[face3.uv_indices[0]].uv[1];
		vbo_manager->vbos[vbo_index][vbo_frame].vertices.push_back( v1 );

		Vertex v2;
		INITVERTEX( v2 );
		v2.location[0] = unique_locations[face3.location_indices[1]].location[0];
		v2.location[1] = unique_locations[face3.location_indices[1]].location[1];
		v2.location[2] = unique_locations[face3.location_indices[1]].location[2];
		v2.normal[0] = unique_normals[face3.normal_indices[1]].normal[0];
		v2.normal[1] = unique_normals[face3.normal_indices[1]].normal[1];
		v2.normal[2] = unique_normals[face3.normal_indices[1]].normal[2];
		v2.texture[0] = unique_uvs[face3.uv_indices[1]].uv[0];
		v2.texture[1] = unique_uvs[face3.uv_indices[1]].uv[1];
		vbo_manager->vbos[vbo_index][vbo_frame].vertices.push_back( v2 );

		Vertex v3;
		INITVERTEX( v3 );
		v3.location[0] = unique_locations[face3.location_indices[2]].location[0];
		v3.location[1] = unique_locations[face3.location_indices[2]].location[1];
		v3.location[2] = unique_locations[face3.location_indices[2]].location[2];
		v3.normal[0] = unique_normals[face3.normal_indices[2]].normal[0];
		v3.normal[1] = unique_normals[face3.normal_indices[2]].normal[1];
		v3.normal[2] = unique_normals[face3.normal_indices[2]].normal[2];
		v3.texture[0] = unique_uvs[face3.uv_indices[2]].uv[0];
		v3.texture[1] = unique_uvs[face3.uv_indices[2]].uv[1];
		vbo_manager->vbos[vbo_index][vbo_frame].vertices.push_back( v3 );
	}

	// Proceed to fill the VBO with data:
	sizes.push_back( 0 );
	ids.push_back( 0 );
	sizes[ sizes.size()-1 ] = vbo_manager->gen_vbo( ids[ ids.size()-1 ], vbo_index, vbo_frame );

	return true;
}
//
//=======================================================================================
//
bool CharacterModel::save_obj( string& filename )
{
	ofstream objFile( filename.c_str(), ios::out | ios::trunc );
	if( objFile.is_open() )
	{
		objFile << "# " << filename << endl;
		objFile << "g " << filename << endl;

		for( unsigned int l = 0; l < unique_locations.size(); l++ )
		{
			Location4 loc = unique_locations[l];
			objFile << "v " << loc.location[0] << " " << loc.location[1] << " " << loc.location[2] << endl;
		}
		objFile << endl;

		for( unsigned int n = 0; n < unique_normals.size(); n++ )
		{
			Normal nor = unique_normals[n];
			objFile << "vn " << nor.normal[0] << " " << nor.normal[1] << " " << nor.normal[2] << endl;
		}
		objFile << endl;

		for( unsigned int u = 0; u < unique_uvs.size(); u++ )
		{
			Uv uv = unique_uvs[u];
			objFile << "vt " << uv.uv[0] << " " << uv.uv[1] << endl;
		}
		objFile << endl;

		for( unsigned int f = 0; f < faces.size(); f++ )
		{
			Face3 face3 = faces[f];
			objFile << "f ";
			objFile << (face3.location_indices[0]+1) << "/" << (face3.uv_indices[0]+1) << "/" << (face3.normal_indices[0]+1);
			objFile << " ";
			objFile << (face3.location_indices[1]+1) << "/" << (face3.uv_indices[1]+1) << "/" << (face3.normal_indices[1]+1);
			objFile << " ";
			objFile << (face3.location_indices[2]+1) << "/" << (face3.uv_indices[2]+1) << "/" << (face3.normal_indices[2]+1);
			objFile << endl;
		}

		objFile.close();
		return true;
	}
	else
	{
		return false;
	}
}
//
//=======================================================================================
//
#ifdef CUDA_PATHS
#ifdef DEMO_SHADER
void CharacterModel::draw_instanced_culled_rigged(	Camera*					cam,
													unsigned int			frame,
													unsigned int			instances,
													unsigned int			_AGENTS_NPOT,
													unsigned int			_ANIMATION_LENGTH,
													unsigned int			_STEP,
													GLuint&					ids_buffer,
													GLuint&					pos_buffer,
													GLuint&					clothing_color_table_tex_id,
													GLuint&					pattern_color_table_tex_id,
													GLuint&					global_mt_tex_id,
													GLuint&					torso_mt_tex_id,
													GLuint&					legs_mt_tex_id,
													GLuint&					rigging_mt_tex_id,
													GLuint&					animation_mt_tex_id,
													GLuint&					facial_mt_tex_id,
													float					lod,
													float					gender,
													float*					viewMat,
													bool					wireframe,
													float					doHandD,
													float					doPatterns,
													float					doColor,
													float					doFacial					)
{
	vbo_manager->render_instanced_culled_rigged_vbo4(	cam,
														ids[ frame ],
														ids_buffer,
														pos_buffer,

														clothing_color_table_tex_id,
														pattern_color_table_tex_id,
														global_mt_tex_id,
														torso_mt_tex_id,
														legs_mt_tex_id,
														rigging_mt_tex_id,
														animation_mt_tex_id,
														facial_mt_tex_id,

														lod,
														gender,
														_AGENTS_NPOT,
														_ANIMATION_LENGTH,
														_STEP,
														sizes[ frame ],
														instances,
														viewMat,
														wireframe,

														doHandD,
														doPatterns,
														doColor,
														doFacial					);
}
//
//=======================================================================================
//
#else
void CharacterModel::draw_instanced_culled_rigged(	Camera*					cam,
													unsigned int			frame,
													unsigned int			instances,
													unsigned int			_AGENTS_NPOT,
													unsigned int			_ANIMATION_LENGTH,
													unsigned int			_STEP,
													GLuint&					posTextureId,
													GLuint&					pos_vbo_id,
													GLuint&					clothing_color_table_tex_id,
													GLuint&					pattern_color_table_tex_id,
													GLuint&					global_mt_tex_id,
													GLuint&					torso_mt_tex_id,
													GLuint&					legs_mt_tex_id,
													GLuint&					rigging_mt_tex_id,
													GLuint&					animation_mt_tex_id,
													GLuint&					facial_mt_tex_id,
													float					lod,
													float					gender,
													float*					viewMat,
													bool					wireframe			)
{
	vbo_manager->render_instanced_culled_rigged_vbo4(	cam,
														ids[ frame ],
														posTextureId,
														pos_vbo_id,

														clothing_color_table_tex_id,
														pattern_color_table_tex_id,
														global_mt_tex_id,
														torso_mt_tex_id,
														legs_mt_tex_id,
														rigging_mt_tex_id,
														animation_mt_tex_id,
														facial_mt_tex_id,

														lod,
														gender,
														_AGENTS_NPOT,
														_ANIMATION_LENGTH,
														_STEP,
														sizes[ frame ],
														instances,
														viewMat,
														wireframe				);
}
#endif
//
//=======================================================================================
//
#else
#ifdef DEMO_SHADER
void CharacterModel::draw_instanced_culled_rigged(	Camera*					cam,
													unsigned int			frame,
													unsigned int			instances,
													unsigned int			_AGENTS_NPOT,
													unsigned int			_ANIMATION_LENGTH,
													unsigned int			_STEP,
													GLuint&					posTextureTarget,
													GLuint&					posTextureId,
													GLuint&					pos_vbo_id,
													GLuint&					clothing_color_table_tex_id,
													GLuint&					pattern_color_table_tex_id,
													GLuint&					global_mt_tex_id,
													GLuint&					torso_mt_tex_id,
													GLuint&					legs_mt_tex_id,
													GLuint&					rigging_mt_tex_id,
													GLuint&					animation_mt_tex_id,
													GLuint&					facial_mt_tex_id,
													float					lod,
													float					gender,
													float*					viewMat,
													bool					wireframe,
													float					doHandD,
													float					doPatterns,
													float					doColor,
													float					doFacial					)
{
	vbo_manager->render_instanced_culled_rigged_vbo2(	cam,
														ids[ frame ],
														posTextureTarget, //MUST BE GL_TEXTURE_RECTANGLE
														posTextureId,
														pos_vbo_id,

														clothing_color_table_tex_id,
														pattern_color_table_tex_id,
														global_mt_tex_id,
														torso_mt_tex_id,
														legs_mt_tex_id,
														rigging_mt_tex_id,
														animation_mt_tex_id,
														facial_mt_tex_id,

														lod,
														gender,
														_AGENTS_NPOT,
														_ANIMATION_LENGTH,
														_STEP,
														sizes[ frame ],
														instances,
														viewMat,
														wireframe,

														doHandD,
														doPatterns,
														doColor,
														doFacial					);
}
#else
//
//=======================================================================================
//
void CharacterModel::draw_instanced_culled_rigged(	Camera*					cam,
													unsigned int			frame,
													unsigned int			instances,
													unsigned int			_AGENTS_NPOT,
													unsigned int			_ANIMATION_LENGTH,
													unsigned int			_STEP,
													GLuint&					posTextureTarget,
													GLuint&					posTextureId,
													GLuint&					pos_vbo_id,
													GLuint&					clothing_color_table_tex_id,
													GLuint&					pattern_color_table_tex_id,
													GLuint&					global_mt_tex_id,
													GLuint&					torso_mt_tex_id,
													GLuint&					legs_mt_tex_id,
													GLuint&					rigging_mt_tex_id,
													GLuint&					animation_mt_tex_id,
													GLuint&					facial_mt_tex_id,
													float					lod,
													float					gender,
													float*					viewMat,
													bool					wireframe			)
{
	vbo_manager->render_instanced_culled_rigged_vbo2(	cam,
														ids[ frame ],
														posTextureTarget, //MUST BE GL_TEXTURE_RECTANGLE
														posTextureId,
														pos_vbo_id,

														clothing_color_table_tex_id,
														pattern_color_table_tex_id,
														global_mt_tex_id,
														torso_mt_tex_id,
														legs_mt_tex_id,
														rigging_mt_tex_id,
														animation_mt_tex_id,
														facial_mt_tex_id,

														lod,
														gender,
														_AGENTS_NPOT,
														_ANIMATION_LENGTH,
														_STEP,
														sizes[ frame ],
														instances,
														viewMat,
														wireframe				);
}
#endif
#endif
//
//=======================================================================================
//
#ifdef CUDA_PATHS
void CharacterModel::draw_instanced_culled_rigged_shadow(	Camera*					cam,
															unsigned int			frame,
															unsigned int			instances,
															unsigned int			_AGENTS_NPOT,
															unsigned int			_ANIMATION_LENGTH,
															unsigned int			_STEP,
															GLuint&					posTextureId,
															GLuint&					pos_vbo_id,
															GLuint&					rigging_mt_tex_id,
															GLuint&					animation_mt_tex_id,
															float					lod,
															float					gender,
															float*					viewMat,
															float*					projMat,
															float*					shadowMat,
															bool					wireframe			)
{
	vbo_manager->render_instanced_culled_rigged_vbo5(	cam,
														ids[ frame ],
														posTextureId,
														pos_vbo_id,

														rigging_mt_tex_id,
														animation_mt_tex_id,

														lod,
														gender,
														_AGENTS_NPOT,
														_ANIMATION_LENGTH,
														_STEP,
														sizes[ frame ],
														instances,
														viewMat,
														projMat,
														shadowMat,
														wireframe				);
}
#else
//
//=======================================================================================
//
void CharacterModel::draw_instanced_culled_rigged_shadow(	Camera*					cam,
															unsigned int			frame,
															unsigned int			instances,
															unsigned int			_AGENTS_NPOT,
															unsigned int			_ANIMATION_LENGTH,
															unsigned int			_STEP,
															GLuint&					posTextureTarget,
															GLuint&					posTextureId,
															GLuint&					pos_vbo_id,
															GLuint&					rigging_mt_tex_id,
															GLuint&					animation_mt_tex_id,
															float					lod,
															float					gender,
															float*					viewMat,
															float*					projMat,
															float*					shadowMat,
															bool					wireframe			)
{
	vbo_manager->render_instanced_culled_rigged_vbo3(	cam,
														ids[ frame ],
														posTextureTarget, //MUST BE GL_TEXTURE_RECTANGLE
														posTextureId,
														pos_vbo_id,

														rigging_mt_tex_id,
														animation_mt_tex_id,

														lod,
														gender,
														_AGENTS_NPOT,
														_ANIMATION_LENGTH,
														_STEP,
														sizes[ frame ],
														instances,
														viewMat,
														projMat,
														shadowMat,
														wireframe				);
}
#endif
//
//=======================================================================================
