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
#ifdef __unix__
	#include "../header/cModel3D.h"
	#include <unistd.h>
#else
	#include "cModel3D.h"
#endif

//=======================================================================================
//
Model3D::Model3D( void )
{
	ppsteps		= 0;
	vbo_frame	= 0;
}
//
//=======================================================================================
//
Model3D::Model3D(	string&			mname,
					string&			rel_path,
					VboManager*		vbo_manager,
					GlslManager*	glsl_manager,
					string&			fname,
					float			scale		)
{
	ppsteps = //aiProcessPreset_TargetRealtime_MaxQuality;
		/*
		aiProcess_RemoveComponent					|
		aiProcess_GenSmoothNormals					|
		aiProcess_CalcTangentSpace					|	// calculate tangents and bitangents if possible
		aiProcess_Triangulate						|	// use only triangles
		aiProcess_SortByPType						|	// use only triangles
		aiProcess_JoinIdenticalVertices				|	// join identical vertices / optimize indexing
		aiProcess_ValidateDataStructure				|	// perform a full validation of the loader's output
		aiProcess_ImproveCacheLocality				| 	// improve the cache locality of the output vertices
		aiProcess_FixInfacingNormals				|	// This step tries to determine which meshes have normal vectors that are facing inwards.
		aiProcess_RemoveRedundantMaterials			|	// remove redundant materials
		aiProcess_FindDegenerates					|	// remove degenerated polygons from the import
		aiProcess_FindInvalidData					|	// detect invalid model data, such as invalid normal vectors
		aiProcess_GenUVCoords						|	// convert spherical, cylindrical, box and planar mapping to proper UVs
		aiProcess_TransformUVCoords					|	// preprocess UV transformations (scaling, translation ...)
		aiProcess_FindInstances						|	// search for instanced meshes and remove them by references to one master
		aiProcess_LimitBoneWeights					|	// limit bone weights to 4 per vertex
		aiProcess_OptimizeMeshes					|	// join small meshes, if possible;
		*/
		0;

	name				= mname;
	this->rel_path		= rel_path;
	fileName			= fname;
	this->vbo_manager	= vbo_manager;
	this->glsl_manager	= glsl_manager;
	this->scale			= scale;
	inited				= false;
	scene				= NULL;
	vbo_frame			= 0;
}
//
//=======================================================================================
//
Model3D::Model3D( Model3D* other )
{
	name				= other->name;
	rel_path			= other->rel_path;
	fileName			= other->fileName;
	vbo_manager			= other->vbo_manager;
	glsl_manager		= other->glsl_manager;
	scale				= other->scale;
	inited				= other->inited;
	if( inited )
	{
		sizes	= other->getSizes();
		ids		= other->getIds();
	}
	scene				= NULL;
	ppsteps				= 0;
	vbo_frame			= 0;
}
//
//=======================================================================================
//
Model3D::~Model3D( void )
{

}
//
//=======================================================================================
//
bool Model3D::init( bool gen_vbos )
{
	string afn = rel_path;
	afn.append( "/" );
	afn.append( fileName );

#ifdef ASSIMP_PURE_C
	//char cCurrentPath[1000];
	//getcwd( cCurrentPath, sizeof(cCurrentPath) );
	//cCurrentPath[sizeof(cCurrentPath) - 1] = '\0';
	//printf ("The current working directory is %s\n", cCurrentPath);
	//printf( "READING: %s\n", afn.c_str() );
	scene = aiImportFile( afn.c_str(), ppsteps );
#else
	importer.SetPropertyInteger( AI_CONFIG_PP_RVC_FLAGS, aiComponent_NORMALS | aiComponent_COLORS | aiComponent_TEXTURES );
	scene = importer.ReadFile( afn.c_str(), ppsteps );
#endif

	if( !scene )
	{
		printf( "ERROR_READING: %s\n", afn.c_str() );
#ifdef ASSIMP_PURE_C
		printf( "%s\n", aiGetErrorString() );
#else
		printf( "%s\n", importer.GetErrorString() );
#endif
		inited = false;
		return inited;
	}

	if( gen_vbos )
	{
		recursive_gather_data_A( scene, scene->mRootNode );
	}
	else
	{
		recursive_gather_data_B( scene, scene->mRootNode );
	}
	inited = true;
	return inited;
}
//
//=======================================================================================
//
void Model3D::recursive_gather_data_A(	const struct aiScene*	sc,
										const struct aiNode*	nd	)
{
	for( unsigned int n = 0; n < nd->mNumMeshes; ++n )
	{
		model_mesh curr_mesh;
		INIT_MODEL_MESH( curr_mesh );
		curr_mesh.vbo_index = vbo_manager->createVBOContainer( fileName, vbo_frame );
		curr_mesh.vbo_frame = vbo_frame;
		vbo_frame++;

		const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];
		for( unsigned int f = 0; f < mesh->mNumFaces; ++f )
		{
			const struct aiFace* face = &mesh->mFaces[f];
			for( unsigned int i = 0; i < face->mNumIndices; i++ )
			{
				unsigned int index = face->mIndices[i];
				Vertex vert;
				INITVERTEX( vert );
				if( mesh->HasTextureCoords( 0 ) )
				{
					vert.texture[0] = mesh->mTextureCoords[0][index].x;
					vert.texture[1] = mesh->mTextureCoords[0][index].y;
				}
				else
				{
					vert.texture[0] = 0.0f;
					vert.texture[1] = 0.0f;
				}
				if( mesh->HasTangentsAndBitangents() )
				{
					vert.tangent[0] = mesh->mTangents[index].x;
					vert.tangent[1] = mesh->mTangents[index].y;
					vert.tangent[2] = mesh->mTangents[index].z;
				}
				if( mesh->mNormals != NULL )
				{
					vert.normal[0] = mesh->mNormals[index].x;
					vert.normal[1] = mesh->mNormals[index].y;
					vert.normal[2] = mesh->mNormals[index].z;
				}
				else
				{
					vert.normal[0] = 0.0f;
					vert.normal[1] = 0.0f;
					vert.normal[2] = 0.0f;
				}
				if( mesh->HasVertexColors( 0 ) )
				{
					vert.color[0] = mesh->mColors[0][index].r;
					vert.color[1] = mesh->mColors[0][index].g;
					vert.color[2] = mesh->mColors[0][index].b;
					vert.color[3] = mesh->mColors[0][index].a;
				}
				else
				{
					vert.color[0] = 1.0f;
					vert.color[1] = 1.0f;
					vert.color[2] = 1.0f;
					vert.color[3] = 1.0f;
				}

				struct aiMaterial *mtl = sc->mMaterials[mesh->mMaterialIndex];
				aiString texPath;
				if( AI_SUCCESS == mtl->GetTexture( aiTextureType_DIFFUSE, 0, &texPath ) )
				{
					if( curr_mesh.tex_id == 0 )
					{
						curr_mesh.tex_file = rel_path + "/" + string(texPath.data);  // get filename
						curr_mesh.tex_id = TextureManager::getInstance()->loadTexture( curr_mesh.tex_file, false, GL_TEXTURE_2D, GL_REPLACE );
					}
				}

				aiColor4D diffuse;
				float transparency;
				if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
				{
					vert.color[0] = diffuse.r;
					vert.color[1] = diffuse.g;
					vert.color[2] = diffuse.b;
					vert.color[3] = diffuse.a;
					if( !curr_mesh.tint_set )
					{
						curr_mesh.tint[0] = diffuse.r;
						curr_mesh.tint[1] = diffuse.g;
						curr_mesh.tint[2] = diffuse.b;
						curr_mesh.tint_set = true;
						if(AI_SUCCESS == mtl->Get(AI_MATKEY_OPACITY,transparency))
						{
							curr_mesh.opacity = transparency;
						}
					}
				}

				aiColor4D ambient;
				if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_AMBIENT, &ambient))
				{

				}
				else
				{

				}

				aiColor4D specular;
				if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &specular))
				{

				}
				else
				{

				}

				aiColor4D emission;
				if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_EMISSIVE, &emission))
				{

				}
				else
				{

				}

				float shininess = 0.0;
				unsigned int max;
				aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS, &shininess, &max);
				{

				}

				vert.location[0] = mesh->mVertices[index].x * scale;
				vert.location[1] = mesh->mVertices[index].y * scale;
				vert.location[2] = mesh->mVertices[index].z * scale;
				vert.location[3] = 1.0f;
				vbo_manager->vbos[curr_mesh.vbo_index][curr_mesh.vbo_frame].vertices.push_back( vert );
			}
		}
		// Proceed to fill the VBO (current mesh) with data:
		curr_mesh.vbo_size = vbo_manager->gen_vbo( curr_mesh.vbo_id, curr_mesh.vbo_index, curr_mesh.vbo_frame );
		meshes.push_back( curr_mesh );
	}
	for( unsigned int n = 0; n < nd->mNumChildren; ++n )
	{
		recursive_gather_data_A( sc, nd->mChildren[n] );
	}
}
//
//=======================================================================================
//
void Model3D::recursive_gather_data_B(	const struct aiScene*	sc,
										const struct aiNode*	nd	)
{
	for( unsigned int n = 0; n < nd->mNumMeshes; ++n )
	{
		const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];
		for( unsigned int f = 0; f < mesh->mNumFaces; ++f )
		{
			Face3 face3;
			INITFACE3( face3 );
			const struct aiFace* face = &mesh->mFaces[f];
			for( unsigned int i = 0; i < face->mNumIndices; i++ )
			{
				unsigned int index = face->mIndices[i];
				Vertex vert;
				INITVERTEX( vert );
				if( mesh->HasTextureCoords( 0 ) )
				{
					vert.texture[0] = mesh->mTextureCoords[0][index].x;
					vert.texture[1] = mesh->mTextureCoords[0][index].y;
				}
				else
				{
					vert.texture[0] = 0.0f;
					vert.texture[1] = 0.0f;
				}
				//->UV_MAP
				float uKey = vert.texture[0] + vert.texture[1];
				unsigned int uvIndex = 0;
				if( u_map.find( uKey ) == u_map.end() )
				{
					u_map.insert( RefMap::value_type( uKey, unique_uvs.size() ) );
					uvIndex = unique_uvs.size();
					Uv uv;
					INITUV( uv );
					uv.uv[0] = vert.texture[0];
					uv.uv[1] = vert.texture[1];
					unique_uvs.push_back( uv );
				}
				else
				{
					bool found = false;
					pair<RefMap::iterator,RefMap::iterator> pair1 = u_map.equal_range( uKey );
					for( ; pair1.first != pair1.second; ++pair1.first )
					{
						if( unique_uvs[pair1.first->second].uv[0] == vert.texture[0] )
						{
							if( unique_uvs[pair1.first->second].uv[1] == vert.texture[1] )
							{
								uvIndex = pair1.first->second;
								found = true;
								break;
							}
						}
					}
					if( found == false )
					{
						u_map.insert( RefMap::value_type( uKey, unique_uvs.size() ) );
						uvIndex = unique_uvs.size();
						Uv uv;
						INITUV( uv );
						uv.uv[0] = vert.texture[0];
						uv.uv[1] = vert.texture[1];
						unique_uvs.push_back( uv );
					}
				}
				indices_uvs.push_back( uvIndex );
				face3.uv_indices[i] = uvIndex;
				//<-UV_MAP
				if( mesh->HasTangentsAndBitangents() )
				{
					vert.tangent[0] = mesh->mTangents[index].x;
					vert.tangent[1] = mesh->mTangents[index].y;
					vert.tangent[2] = mesh->mTangents[index].z;
				}
				if( mesh->mNormals != NULL )
				{
					vert.normal[0] = mesh->mNormals[index].x;
					vert.normal[1] = mesh->mNormals[index].y;
					vert.normal[2] = mesh->mNormals[index].z;
				}
				else
				{
					vert.normal[0] = 0.0f;
					vert.normal[1] = 0.0f;
					vert.normal[2] = 0.0f;
				}
				//->NORMAL_MAP
				float nKey = vert.normal[0] + vert.normal[1] + vert.normal[2];
				unsigned int normalIndex = 0;
				if( n_map.find( nKey ) == n_map.end() )
				{
					n_map.insert( RefMap::value_type( nKey, unique_normals.size() ) );
					normalIndex = unique_normals.size();
					Normal normal;
					INITNORMAL( normal );
					normal.normal[0] = vert.normal[0];
					normal.normal[1] = vert.normal[1];
					normal.normal[2] = vert.normal[2];
					unique_normals.push_back( normal );
				}
				else
				{
					bool found = false;
					pair<RefMap::iterator,RefMap::iterator> pair1 = n_map.equal_range( nKey );
					for( ; pair1.first != pair1.second; ++pair1.first )
					{
						if( unique_normals[pair1.first->second].normal[0] == vert.normal[0] )
						{
							if( unique_normals[pair1.first->second].normal[1] == vert.normal[1] )
							{
								if( unique_normals[pair1.first->second].normal[2] == vert.normal[2] )
								{
									normalIndex = pair1.first->second;
									found = true;
									break;
								}
							}
						}
					}
					if( found == false )
					{
						n_map.insert( RefMap::value_type( nKey, unique_normals.size() ) );
						normalIndex = unique_normals.size();
						Normal normal;
						INITNORMAL( normal );
						normal.normal[0] = vert.normal[0];
						normal.normal[1] = vert.normal[1];
						normal.normal[2] = vert.normal[2];
						unique_normals.push_back( normal );
					}
				}
				indices_normals.push_back( normalIndex );
				face3.normal_indices[i] = normalIndex;
				//<-NORMAL_MAP
				if( mesh->HasVertexColors( 0 ) )
				{
					vert.color[0] = mesh->mColors[0][index].r;
					vert.color[1] = mesh->mColors[0][index].g;
					vert.color[2] = mesh->mColors[0][index].b;
					vert.color[3] = mesh->mColors[0][index].a;
				}
				else
				{
					vert.color[0] = 1.0f;
					vert.color[1] = 1.0f;
					vert.color[2] = 1.0f;
					vert.color[3] = 1.0f;
				}

				struct aiMaterial *mtl = sc->mMaterials[mesh->mMaterialIndex];

				aiString texPath;
				if( AI_SUCCESS == mtl->GetTexture( aiTextureType_DIFFUSE, 0, &texPath ) )
				{

				}

				aiColor4D diffuse;
				if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
				{
					vert.color[0] = diffuse.r;
					vert.color[1] = diffuse.g;
					vert.color[2] = diffuse.b;
					vert.color[3] = diffuse.a;
				}

				aiColor4D ambient;
				if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_AMBIENT, &ambient))
				{

				}
				else
				{

				}

				aiColor4D specular;
				if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &specular))
				{

				}
				else
				{

				}

				aiColor4D emission;
				if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_EMISSIVE, &emission))
				{

				}
				else
				{

				}

				float shininess = 0.0;
				unsigned int max;
				aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS, &shininess, &max);
				{

				}

				vert.location[0] = mesh->mVertices[index].x * scale;
				vert.location[1] = mesh->mVertices[index].y * scale;
				vert.location[2] = mesh->mVertices[index].z * scale;
				vert.location[3] = 1.0f;

				//->LOCATIONS_MAP
				float lKey = vert.location[0] + vert.location[1] + vert.location[2];
				unsigned int locationIndex = 0;
				if( l_map.find( lKey ) == l_map.end() )
				{
					l_map.insert( RefMap::value_type( lKey, unique_locations.size() ) );
					locationIndex = unique_locations.size();
					Location4 location;
					INITLOCATION4( location );
					location.location[0] = vert.location[0];
					location.location[1] = vert.location[1];
					location.location[2] = vert.location[2];
					location.location[3] = vert.location[3];
					unique_locations.push_back( location );
				}
				else
				{
					bool found = false;
					pair<RefMap::iterator,RefMap::iterator> pair1 = l_map.equal_range( lKey );
					for( ; pair1.first != pair1.second; ++pair1.first )
					{
						if( unique_locations[pair1.first->second].location[0] == vert.location[0] )
						{
							if( unique_locations[pair1.first->second].location[1] == vert.location[1] )
							{
								if( unique_locations[pair1.first->second].location[2] == vert.location[2] )
								{
									locationIndex = pair1.first->second;
									found = true;
									break;
								}
							}
						}
					}
					if( found == false )
					{
						l_map.insert( RefMap::value_type( lKey, unique_locations.size() ) );
						locationIndex = unique_locations.size();
						Location4 location;
						INITLOCATION4( location );
						location.location[0] = vert.location[0];
						location.location[1] = vert.location[1];
						location.location[2] = vert.location[2];
						location.location[3] = vert.location[3];
						unique_locations.push_back( location );
					}
				}
				indices_locations.push_back( locationIndex );
				face3.location_indices[i] = locationIndex;
				//<-LOCATIONS_MAP
			}
			faces.push_back( face3 );
		}
	}
	for( unsigned int n = 0; n < nd->mNumChildren; ++n )
	{
		recursive_gather_data_B( sc, nd->mChildren[n] );
	}
}
//
//=======================================================================================
//
vector<GLuint>& Model3D::getSizes( void )
{
	return sizes;
}
//
//=======================================================================================
//
vector<GLuint>& Model3D::getIds( void )
{
	return ids;
}
//
//=======================================================================================
//
vector<unsigned int>& Model3D::getIndicesUvs( void )
{
	return indices_uvs;
}
//
//=======================================================================================
//
vector<unsigned int>& Model3D::getIndicesNormals( void )
{
	return indices_normals;
}
//
//=======================================================================================
//
vector<unsigned int>& Model3D::getIndicesLocations( void )
{
	return indices_locations;
}
//
//=======================================================================================
//
vector<Location4>& Model3D::getUniqueLocations( void )
{
	return unique_locations;
}
//
//=======================================================================================
//
vector<Normal>& Model3D::getUniqueNormals( void )
{
	return unique_normals;
}
//
//=======================================================================================
//
vector<Uv>& Model3D::getUniqueUvs( void )
{
	return unique_uvs;
}
//
//=======================================================================================
//
vector<Face3>& Model3D::getFaces( void )
{
	return faces;
}
//
//=======================================================================================
//
vector<MODEL_MESH>& Model3D::getMeshes( void )
{
	return meshes;
}
//
//=======================================================================================
