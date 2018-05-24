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
#include "cCharacterGroup.h"

//=======================================================================================
//
CharacterGroup::CharacterGroup( LogManager*		log_manager_,
								GlErrorManager*	err_manager_,
								string&			_name,
								MODEL_GENDER	_gender,
								MODEL_TYPE		_type,
								ModelProps*		mProps,
								ModelProps*		mFacialProps,
								ModelProps*		mRiggingProps		)
{
	name					= _name;
	gender					= _gender;
	type					= _type;
	model_props				= mProps;
	facial_props			= mFacialProps;
	rigging_props			= mRiggingProps;
	log_manager				= log_manager_;
	err_manager				= err_manager_;

	clothingColorTableId	= 0;
	patternColorTableId		= 0;
	if( gender == MG_FEMALE )
	{
		string pattern_name		= "assets/female/rgba/female_patterns.rgba";
		patternColorTableId		= TextureManager::getInstance()->loadRawTexture(	pattern_name,
																					GL_NEAREST,
																					1,
																					GL_RGBA32F,
																					GL_RGBA,
																					GL_TEXTURE_RECTANGLE );
		string clothing_name	= "assets/female/rgba/female_color_table.rgba";
		clothingColorTableId	= TextureManager::getInstance()->loadRawTexture(	clothing_name,
																					GL_NEAREST,
																					11,
																					GL_RGBA32F,
																					GL_RGBA,
																					GL_TEXTURE_RECTANGLE );
	}
	else if( gender == MG_MALE )
	{
		string pattern_name		= "assets/male/rgba/male_patterns.rgba";
		patternColorTableId		= TextureManager::getInstance()->loadRawTexture(	pattern_name,
																					GL_NEAREST,
																					1,
																					GL_RGBA32F,
																					GL_RGBA,
																					GL_TEXTURE_RECTANGLE );
		string clothing_name	= "assets/male/rgba/male_color_table.rgba";
		clothingColorTableId	= TextureManager::getInstance()->loadRawTexture(	clothing_name,
																					GL_NEAREST,
																					11,
																					GL_RGBA32F,
																					GL_RGBA,
																					GL_TEXTURE_RECTANGLE );
	}

	globalMtId				= 0;
	torsoMtId				= 0;
	legsMtId				= 0;
	combinedMtId			= 0;

	facialMtId				= 0;
	riggingMtId				= 0;
}
//
//=======================================================================================
//
CharacterGroup::~CharacterGroup( void )
{

}
//
//=======================================================================================
//
void CharacterGroup::addAnimation(	string	_animation,
									GLuint	_frames,
									GLuint	_duration	)
{
	animation			= _animation;
	animation_frames	= _frames;
	animation_duration	= _duration;
	Animation* anim		= new Animation();
	string ext			= "rgba";
	string frameName	= "frame";
	string restName		= "";
	if( gender == MG_FEMALE )
	{
		restName = "woman_rest";
		anim->loadMultitexture( animation, ext, restName, frameName, 0, animation_frames );
	}
	else if( gender == MG_MALE )
	{
		restName = "man_rest";
		anim->loadMultitexture( animation, ext, restName, frameName, 0, animation_frames );
	}
	animations.push_back( anim );
}
//
//=======================================================================================
//
ModelProps* CharacterGroup::getModelProps( void )
{
	return model_props;
}
//
//=======================================================================================
//
void CharacterGroup::setOutfit( string ref_head,
							    string ref_hair,
								string ref_torso,
								string ref_legs,
								string ref_palette )
{
	ref_outfit_head		= ref_head;
	ref_outfit_hair		= ref_hair;
	ref_outfit_torso	= ref_torso;
	ref_outfit_legs		= ref_legs;
	outfit_palette		= ref_palette;
}
//
//=======================================================================================
//
void CharacterGroup::setWeight( float fat,
								float average,
								float thin,
								float strong )
{
	if( abs( 1.0f - (fat + average + thin + strong) ) <= 0.01f )
	{
		weight_fat = fat;
		weight_average = average;
		weight_thin = thin;
		weight_strong = strong;
	}
}
//
//=======================================================================================
//
void CharacterGroup::setHeight( float min,
								float max )
{
	if( min < max )
	{
		height_min = min;
		height_max = max;
	}
}
//
//=======================================================================================
//
void CharacterGroup::setFace(	string		rf_wrinkles,
								string		rf_eye_sockets,
								string		rf_spots,
								string		rf_beard,
								string		rf_moustache,
								string		rf_makeup		)
{
	ref_facial_wrinkles		= rf_wrinkles;
	ref_facial_eye_sockets	= rf_eye_sockets;
	ref_facial_spots		= rf_spots;
	ref_facial_beard		= rf_beard;
	ref_facial_moustache	= rf_moustache;
	ref_facial_makeup		= rf_makeup;
}
//
//=======================================================================================
//
void CharacterGroup::setRig(	string		rr_zones,
								string		rr_weights,
								string		rr_displacement,
								string		rr_animation	)
{
	ref_rigging_zones			= rr_zones;
	ref_rigging_weights			= rr_weights;
	ref_rigging_displacement	= rr_displacement;
	ref_rigging_animation		= rr_animation;
}
//
//=======================================================================================
//
bool CharacterGroup::genCharacterModel( VboManager* vbo_manager, GlslManager* glsl_manager, float scale )
{
	genClothingMultitextures();
	genFacialMultitexture();
	genRiggingMultitexture();

	ModelProp* mpHead	= model_props->getProp( ref_outfit_head		);
	ModelProp* mpHair	= NULL;
	if( ref_outfit_hair.length() > 0 )
	{
		mpHair = model_props->getProp( ref_outfit_hair );
	}
	ModelProp* mpTorso	= model_props->getProp( ref_outfit_torso	);
	ModelProp* mpLegs	= model_props->getProp( ref_outfit_legs		);

	for( unsigned int LOD = 0; LOD < NUM_LOD; LOD++ )
	{
		LOD_TYPE lod_type = (LOD_TYPE)LOD;

		if( lod_type == LOD_HI )
		{
			Model3D* mHead	= NULL;
			Model3D* mHair	= NULL;
			Model3D* mTorso	= NULL;
			Model3D* mLegs	= NULL;
			mHead			= mpHead->model3D;
			if( mpHair )
			{
				mHair		= mpHair->model3D;
			}
			mTorso			= mpTorso->model3D;
			mLegs			= mpLegs->model3D;
			character_model[LOD] = new CharacterModel(	lod_type,
														name,
														vbo_manager,
														glsl_manager,
														mHead,
														mHair,
														mTorso,
														mLegs		);
		}
		else if( lod_type == LOD_ME )
		{
			Model3D* mHead				= NULL;
			Model3D* mHair				= NULL;
			Model3D* mTorso				= NULL;
			Model3D* mLegs				= NULL;

			string lod_me_head_file		= "";
			string lod_me_torso_file	= "";
			string lod_me_legs_file		= "";

			string rel_head_path		= "";
			string rel_torso_path		= "";
			string rel_legs_path		= "";

			string hi_head_file			= mpHead->file;
			string hi_legs_file			= mpLegs->file;
			string hi_torso_file		= mpTorso->file;

			vector<string> sep_head_1	= StringUtils::split( hi_head_file, '/' );
			string base_head_name		= sep_head_1[sep_head_1.size()-1];
			vector<string> sep_head_2	= StringUtils::split( base_head_name, '_' );

			vector<string> sep_torso_1	= StringUtils::split( hi_torso_file, '/' );
			string base_torso_name		= sep_torso_1[sep_torso_1.size()-1];
			vector<string> sep_torso_2	= StringUtils::split( base_torso_name, '_' );

			vector<string> sep_legs_1	= StringUtils::split( hi_legs_file, '/' );
			string base_legs_name		= sep_legs_1[sep_legs_1.size()-1];
			vector<string> sep_legs_2	= StringUtils::split( base_legs_name, '_' );

			for( unsigned int s1 = 0; s1 < (sep_head_1.size()-1); s1++ )
			{
				rel_head_path.append( sep_head_1[s1] );
				if( (s1+1) < (sep_head_1.size()-1) )
				{
					rel_head_path.append( "/" );
				}
			}
			for( unsigned int s2 = 0; s2 < (sep_head_2.size()-1); s2++ )
			{
				lod_me_head_file.append( sep_head_2[s2] );
				lod_me_head_file.append( "_" );
			}
			lod_me_head_file.append( "medium.obj" );

			for( unsigned int s1 = 0; s1 < (sep_torso_1.size()-1); s1++ )
			{
				rel_torso_path.append( sep_torso_1[s1] );
				if( (s1+1) < (sep_torso_1.size()-1) )
				{
					rel_torso_path.append( "/" );
				}
			}
			for( unsigned int s2 = 0; s2 < (sep_torso_2.size()-1); s2++ )
			{
				lod_me_torso_file.append( sep_torso_2[s2] );
				lod_me_torso_file.append( "_" );
			}
			lod_me_torso_file.append( "medium.obj" );

			for( unsigned int s1 = 0; s1 < (sep_legs_1.size()-1); s1++ )
			{
				rel_legs_path.append( sep_legs_1[s1] );
				if( (s1+1) < (sep_legs_1.size()-1) )
				{
					rel_legs_path.append( "/" );
				}
			}
			for( unsigned int s2 = 0; s2 < (sep_legs_2.size()-1); s2++ )
			{
				lod_me_legs_file.append( sep_legs_2[s2] );
				lod_me_legs_file.append( "_" );
			}
			lod_me_legs_file.append( "medium.obj" );

			mHead	= new Model3D(	mpHead->name,
									rel_head_path,
									vbo_manager,
									glsl_manager,
									lod_me_head_file,
									scale			);
			if( mHead->init( false ) == false )
			{
				return false;
			}

			if( mpHair )
			{
				mHair = mpHair->model3D;
			}

			mTorso	= new Model3D(	mpTorso->name,
									rel_torso_path,
									vbo_manager,
									glsl_manager,
									lod_me_torso_file,
									scale			);
			if( mTorso->init( false ) == false )
			{
				return false;
			}

			mLegs	= new Model3D(	mpLegs->name,
									rel_legs_path,
									vbo_manager,
									glsl_manager,
									lod_me_legs_file,
									scale			);
			if( mLegs->init( false ) == false )
			{
				return false;
			}

			character_model[LOD] = new CharacterModel(	lod_type,
														name,
														vbo_manager,
														glsl_manager,
														mHead,
														mHair,
														mTorso,
														mLegs		);
		}
		else if( lod_type == LOD_LO )
		{
			string low_res_file = "";
			if( type == MT_HUMAN )
			{
				low_res_file = string( "low_res.obj" );
			}
			else if( type == MT_LEMMING )
			{
				low_res_file = string( "lemming_low_res.obj" );
			}
			string low_res_rel_path = string( "assets/low_res" );
			Model3D* lowRes = new Model3D( name,
										   low_res_rel_path,
										   vbo_manager,
										   glsl_manager,
										   low_res_file,
										   scale			);
			if( lowRes->init( false ) == false )
			{
				return false;
			}

			character_model[LOD] = new CharacterModel(	lod_type,
														name,
														vbo_manager,
														glsl_manager,
														lowRes		);
		}

		string lod_str = static_cast<ostringstream*>( &(ostringstream() << LOD) )->str();
		if( character_model[LOD]->stitch_parts() )
		{
			string obj_name = string( "assets/stitched/" );
			obj_name.append( name );
			obj_name.append( "_lod_" );
			obj_name.append( lod_str );
			obj_name.append( ".obj" );
			if( character_model[LOD]->save_obj( obj_name ) == false )
			{
				return false;
			}
		}
		else
		{
			return false;
		}
	}
	return true;
}
//
//=======================================================================================
//
void CharacterGroup::nextFrame( void )
{
	for( unsigned int a = 0; a < animations.size(); a++ )
	{
		animations[a]->nextFrame();
	}
}
//
//=======================================================================================
//
string& CharacterGroup::getName( void )
{
	return name;
}
//
//=======================================================================================
//
void CharacterGroup::genClothingMultitextures( void )
{
	ModelProp* mpHead	= model_props->getProp( ref_outfit_head		);
	ModelProp* mpHair	= model_props->getProp( ref_outfit_hair		);
	ModelProp* mpTorso	= model_props->getProp( ref_outfit_torso	);
	ModelProp* mpLegs	= model_props->getProp( ref_outfit_legs		);

	globalMtNames.clear();
	globalMtParameters.clear();

	torsoMtNames.clear();
	torsoMtParameters.clear();

	legsMtNames.clear();
	legsMtParameters.clear();

	combinedMtNames.clear();
	combinedMtParameters.clear();

//->SKIN_ATLAS
	bool skin_atlas_found = false;
	for( unsigned int h = 0; h < mpHead->ref_atlas.size(); h++ )
	{
		ModelProp* temp = mpHead->ref_atlas[h];
		if( temp->type == PP_ATLAS && temp->subtype == PST_SKIN )
		{
			globalMtNames.push_back( temp->file );
			globalMtParameters.push_back( GL_NEAREST );
			skin_atlas_found = true;
			break;
		}
	}
	if( skin_atlas_found == false )
	{
		for( unsigned int h = 0; h < mpTorso->ref_atlas.size(); h++ )
		{
			ModelProp* temp = mpTorso->ref_atlas[h];
			if( temp->type == PP_ATLAS && temp->subtype == PST_SKIN )
			{
				globalMtNames.push_back( temp->file );
				globalMtParameters.push_back( GL_NEAREST );
				skin_atlas_found = true;
				break;
			}
		}
	}
	if( skin_atlas_found == false )
	{
		for( unsigned int h = 0; h < mpLegs->ref_atlas.size(); h++ )
		{
			ModelProp* temp = mpLegs->ref_atlas[h];
			if( temp->type == PP_ATLAS && temp->subtype == PST_SKIN )
			{
				globalMtNames.push_back( temp->file );
				globalMtParameters.push_back( GL_NEAREST );
				skin_atlas_found = true;
				break;
			}
		}
	}
	if( skin_atlas_found == false )
	{
		globalMtNames.push_back( "assets/blank.dds" );
		globalMtParameters.push_back( GL_NEAREST );
	}
//<-SKIN_ATLAS

//->HAIR_ATLAS
	bool hair_atlas_found = false;
	for( unsigned int h = 0; h < mpHead->ref_atlas.size(); h++ )
	{
		ModelProp* temp = mpHead->ref_atlas[h];
		if( temp->type == PP_ATLAS && temp->subtype == PST_HAIR )
		{
			globalMtNames.push_back( temp->file );
			globalMtParameters.push_back( GL_NEAREST );
			hair_atlas_found = true;
			break;
		}
	}
	if( hair_atlas_found == false && mpHair )
	{
		for( unsigned int h = 0; h < mpHair->ref_atlas.size(); h++ )
		{
			ModelProp* temp = mpHair->ref_atlas[h];
			if( temp->type == PP_ATLAS && temp->subtype == PST_HAIR )
			{
				globalMtNames.push_back( temp->file );
				globalMtParameters.push_back( GL_NEAREST );
				hair_atlas_found = true;
				break;
			}
		}
	}
	if( hair_atlas_found == false )
	{
		for( unsigned int h = 0; h < mpTorso->ref_atlas.size(); h++ )
		{
			ModelProp* temp = mpTorso->ref_atlas[h];
			if( temp->type == PP_ATLAS && temp->subtype == PST_HAIR )
			{
				globalMtNames.push_back( temp->file );
				globalMtParameters.push_back( GL_NEAREST );
				hair_atlas_found = true;
				break;
			}
		}
	}
	if( hair_atlas_found == false )
	{
		for( unsigned int h = 0; h < mpLegs->ref_atlas.size(); h++ )
		{
			ModelProp* temp = mpLegs->ref_atlas[h];
			if( temp->type == PP_ATLAS && temp->subtype == PST_HAIR )
			{
				globalMtNames.push_back( temp->file );
				globalMtParameters.push_back( GL_NEAREST );
				hair_atlas_found = true;
				break;
			}
		}
	}
	if( hair_atlas_found == false )
	{
		globalMtNames.push_back( "assets/blank.dds" );
		globalMtParameters.push_back( GL_NEAREST );
	}
//<-HAIR_ATLAS

//->CAP_ATLAS
	bool cap_atlas_found = false;
	for( unsigned int h = 0; h < mpHead->ref_atlas.size(); h++ )
	{
		ModelProp* temp = mpHead->ref_atlas[h];
		if( temp->type == PP_ATLAS && temp->subtype == PST_CAP )
		{
			globalMtNames.push_back( temp->file );
			globalMtParameters.push_back( GL_NEAREST );
			cap_atlas_found = true;
			break;
		}
	}
	if( cap_atlas_found == false )
	{
		globalMtNames.push_back( "assets/blank.dds" );
		globalMtParameters.push_back( GL_NEAREST );
	}
//<-CAP_ATLAS

//->WRINKLES
	bool torso_wrinkles_found = false;
	bool legs_wrinkles_found = false;
	if( mpTorso->ref_wrinkles )
	{
		if( mpTorso->ref_wrinkles->file.length() > 4 )
		{
			torsoMtNames.push_back( mpTorso->ref_wrinkles->file );
			torsoMtParameters.push_back( GL_NEAREST );
			torso_wrinkles_found = true;
		}
	}
	if( torso_wrinkles_found == false )
	{
		torsoMtNames.push_back( "assets/blank.tga" );
		torsoMtParameters.push_back( GL_NEAREST );
	}
	if( mpLegs->ref_wrinkles )
	{
		if( mpLegs->ref_wrinkles->file.length() > 4 )
		{
			legsMtNames.push_back( mpLegs->ref_wrinkles->file );
			legsMtParameters.push_back( GL_NEAREST );
			legs_wrinkles_found = true;
		}
	}
	if( legs_wrinkles_found == false )
	{
		legsMtNames.push_back( "assets/blank.tga" );
		legsMtParameters.push_back( GL_NEAREST );
	}
//<-WRINKLES

//->PATTERNS
	bool legs_pattern_found = false;
	bool torso_pattern_found = false;
	for( unsigned int t = 0; t < mpTorso->ref_pattern.size(); t++ )
	{
		ModelProp* temp = mpTorso->ref_pattern[t];
		if( temp->type == PP_PATTERN )
		{
			torsoMtNames.push_back( temp->file );
			torsoMtParameters.push_back( GL_NEAREST );
			torso_pattern_found = true;
		}
	}
	if( torso_pattern_found == false )
	{
		torsoMtNames.push_back( "assets/blank.tga" );
		torsoMtParameters.push_back( GL_NEAREST );
	}
	for( unsigned int l = 0; l < mpLegs->ref_pattern.size(); l++ )
	{
		ModelProp* temp = mpLegs->ref_pattern[l];
		if( temp->type == PP_PATTERN )
		{
			legsMtNames.push_back( temp->file );
			legsMtParameters.push_back( GL_NEAREST );
			legs_pattern_found = true;
		}
	}
	if( legs_pattern_found == false )
	{
		legsMtNames.push_back( "assets/blank.tga" );
		legsMtParameters.push_back( GL_NEAREST );
	}
//<-PATTERNS

	globalMtId = TextureManager::getInstance()->loadTexture3D(	globalMtNames,
																globalMtParameters,
																false,
																GL_TEXTURE_2D_ARRAY	);
	torsoMtId = TextureManager::getInstance()->loadTexture3D(	torsoMtNames,
																torsoMtParameters,
																false,
																GL_TEXTURE_2D_ARRAY	);
	legsMtId = TextureManager::getInstance()->loadTexture3D(	legsMtNames,
																legsMtParameters,
																false,
																GL_TEXTURE_2D_ARRAY	);

	log_manager->log( LogManager::INFORMATION, "%s CLOTHING MULTITEXTURES:", name.c_str() );
	log_manager->log( LogManager::INFORMATION, "\tGLOBAL:" );
	for( unsigned int g = 0; g < globalMtNames.size(); g++ )
	{
		string name;
		StringUtils::getNameFromPath( (char*)globalMtNames[g].c_str(), name );
		log_manager->log( LogManager::INFORMATION, "\t\t[%d]\t%s", g, name.c_str() );
	}

	log_manager->log( LogManager::INFORMATION, "\tTORSO:" );
	for( unsigned int t = 0; t < torsoMtNames.size(); t++ )
	{
		string name;
		StringUtils::getNameFromPath( (char*)torsoMtNames[t].c_str(), name );
		log_manager->log( LogManager::INFORMATION, "\t\t[%d]\t%s", t, name.c_str() );
	}

	log_manager->log( LogManager::INFORMATION, "\tLEGS:" );
	for( unsigned int l = 0; l < legsMtNames.size(); l++ )
	{
		string name;
		StringUtils::getNameFromPath( (char*)legsMtNames[l].c_str(), name );
		log_manager->log( LogManager::INFORMATION, "\t\t[%d]\t%s", l, name.c_str() );
	}
}
//
//=======================================================================================
//
void CharacterGroup::genFacialMultitexture( void )
{
	ModelProp* mpWrinkles		= facial_props->getProp( ref_facial_wrinkles	);
	ModelProp* mpEye_sockets	= facial_props->getProp( ref_facial_eye_sockets	);
	ModelProp* mpSpots			= facial_props->getProp( ref_facial_spots		);
	ModelProp* mpBeard			= facial_props->getProp( ref_facial_beard		);
	ModelProp* mpMoustache		= facial_props->getProp( ref_facial_moustache	);
	ModelProp* mpMakeup			= facial_props->getProp( ref_facial_makeup		);

	facialMtNames.clear();

	if( mpWrinkles->file.length() > 4 )
	{
		facialMtNames.push_back( mpWrinkles->file );
		facialMtParameters.push_back( GL_LINEAR );
	}
	if( mpBeard->file.length() > 4 )
	{
		facialMtNames.push_back( mpBeard->file );
		facialMtParameters.push_back( GL_LINEAR );
	}
	if( mpMakeup->file.length() > 4 )
	{
		facialMtNames.push_back( mpMakeup->file );
		facialMtParameters.push_back( GL_LINEAR );
	}

	facialMtId = TextureManager::getInstance()->loadTexture3D(	facialMtNames,
																facialMtParameters,
																false,
																GL_TEXTURE_2D_ARRAY	);

	if( mpWrinkles->file.length() > 4 )
	{
		mpWrinkles->loaded = true;
	}
	if( mpBeard->file.length() > 4 )
	{
		mpBeard->loaded = true;
	}
	if( mpMakeup->file.length() > 4 )
	{
		mpMakeup->loaded = true;
	}

	log_manager->log( LogManager::INFORMATION, "%s FACIAL MULTITEXTURES:", name.c_str() );
	for( unsigned int f = 0; f < facialMtNames.size(); f++ )
	{
		string name;
		StringUtils::getNameFromPath( (char*)facialMtNames[f].c_str(), name );
		log_manager->log( LogManager::INFORMATION, "\t[%d]\t%s", f, name.c_str() );
	}
}
//
//=======================================================================================
//
void CharacterGroup::genRiggingMultitexture( void )
{
	ModelProp* mpZones			= rigging_props->getProp( ref_rigging_zones			);
	ModelProp* mpWeights		= rigging_props->getProp( ref_rigging_weights		);
	ModelProp* mpDisplacement	= rigging_props->getProp( ref_rigging_displacement	);

	riggingMtNames.clear();

	if( mpZones->file.length() > 4 )
	{
		riggingMtNames.push_back( mpZones->file );
		riggingMtParameters.push_back( GL_NEAREST );
	}
	if( mpWeights->file.length() > 4 )
	{
		riggingMtNames.push_back( mpWeights->file );
		riggingMtParameters.push_back( GL_NEAREST );
	}
	if( mpDisplacement->file.length() > 4 )
	{
		riggingMtNames.push_back( mpDisplacement->file );
		riggingMtParameters.push_back( GL_NEAREST );
	}

	riggingMtId = TextureManager::getInstance()->loadTexture3D(	riggingMtNames,
																riggingMtParameters,
																false,
																GL_TEXTURE_2D_ARRAY	);

	if( mpZones->file.length() > 4 )
	{
		mpZones->loaded = true;
	}
	if( mpWeights->file.length() > 4 )
	{
		mpWeights->loaded = true;
	}
	if( mpDisplacement->file.length() > 4 )
	{
		mpDisplacement->loaded = true;
	}

	log_manager->log( LogManager::INFORMATION, "%s RIGGING MULTITEXTURES:", name.c_str() );
	for( unsigned int r = 0; r < riggingMtNames.size(); r++ )
	{
		string name;
		StringUtils::getNameFromPath( (char*)riggingMtNames[r].c_str(), name );
		log_manager->log( LogManager::INFORMATION, "\t[%d]\t%s", r, name.c_str() );
	}
}
//
//=======================================================================================
//
#ifdef CUDA_PATHS
#ifdef DEMO_SHADER
void CharacterGroup::draw_instanced_culled_rigged	(	Camera*					cam,
														unsigned int			frame,
														unsigned int			_AGENTS_NPOT,
														struct sVBOLod*			vbo_lods,
														GLuint&					posTextureBuffer,
														float					dt,
														float*					viewMat,
														float*					projMat,
														float*					shadowMat,
														bool					wireframe,
														bool					shadows,
														bool					doHandD,
														bool					doPatterns,
														bool					doColor,
														bool					doFacial		)
{
	//err_manager->getError( "BEGIN : CharacterGroup::draw_instanced_culled_rigged" );
	for( unsigned int LOD = 0; LOD < NUM_LOD; LOD++ )
	{
		float dH = doHandD?1.0f:0.0f;
		float dP = doPatterns?1.0f:0.0f;
		float dC = doColor?1.0f:0.0f;
		float dF = doFacial?1.0f:0.0f;
		character_model[LOD]->draw_instanced_culled_rigged(	cam,
															frame,
															vbo_lods[LOD].primitivesWritten,
															_AGENTS_NPOT,
															animations[0]->getLength(),
															animations[0]->step,
															vbo_lods[LOD].id,
															posTextureBuffer,
															clothingColorTableId,
															patternColorTableId,
															globalMtId,
															torsoMtId,
															legsMtId,
															riggingMtId,
															animations[0]->multitextureID,
															facialMtId,
															(float)LOD,		// Originalmente dt.
															(float)gender,
															viewMat,
															wireframe,
															dH,
															dP,
															dC,
															dF								);
		//printf( "[%d]: %03d\t", LOD, vbo_lods[LOD].primitivesWritten );
		if( shadows )
		{
			character_model[LOD_LO]->draw_instanced_culled_rigged_shadow(	cam,
																			frame,
																			vbo_lods[LOD].primitivesWritten,
																			_AGENTS_NPOT,
																			animations[0]->getLength(),
																			animations[0]->step,
																			posTextureBuffer,
																			vbo_lods[LOD].id,
																			riggingMtId,
																			animations[0]->multitextureID,
																			(float)LOD_LO,		// Originalmente dt.
																			(float)gender,
																			viewMat,
																			projMat,
																			shadowMat,
																			wireframe						);
		}
	}
	//printf( "\n" );
	//err_manager->getError( "END : CharacterGroup::draw_instanced_culled_rigged" );
}
#else
//
//=======================================================================================
//
void CharacterGroup::draw_instanced_culled_rigged	(	Camera*					cam,
														unsigned int			frame,
														unsigned int			_AGENTS_NPOT,
														struct sVBOLod*			vbo_lods,
														GLuint&					posTextureId,
														float					dt,
														float*					viewMat,
														float*					projMat,
														float*					shadowMat,
														bool					wireframe,
														bool					shadows			)
{
	//err_manager->getError( "BEGIN : CharacterGroup::draw_instanced_culled_rigged" );
	for( unsigned int LOD = 0; LOD < NUM_LOD; LOD++ )
	{
		character_model[LOD]->draw_instanced_culled_rigged(	cam,
															frame,
															vbo_lods[LOD].primitivesWritten,
															_AGENTS_NPOT,
															animations[0]->getLength(),
															animations[0]->step,
															posTextureId,
															vbo_lods[LOD].id,
															clothingColorTableId,
															patternColorTableId,
															globalMtId,
															torsoMtId,
															legsMtId,
															riggingMtId,
															animations[0]->multitextureID,
															facialMtId,
															(float)LOD,		// Originalmente dt.
															(float)gender,
															viewMat,
															wireframe						);
		//printf( "[%d]: %03d\t", LOD, vbo_lods[LOD].primitivesWritten );
		if( shadows )
		{
			character_model[LOD_LO]->draw_instanced_culled_rigged_shadow(	cam,
																			frame,
																			vbo_lods[LOD].primitivesWritten,
																			_AGENTS_NPOT,
																			animations[0]->getLength(),
																			animations[0]->step,
																			posTextureId,
																			vbo_lods[LOD].id,
																			riggingMtId,
																			animations[0]->multitextureID,
																			(float)LOD_LO,		// Originalmente dt.
																			(float)gender,
																			viewMat,
																			projMat,
																			shadowMat,
																			wireframe						);
		}
	}
	//printf( "\n" );
	//err_manager->getError( "END : CharacterGroup::draw_instanced_culled_rigged" );
}
#endif
#else
#ifdef DEMO_SHADER
//
//=======================================================================================
//
void CharacterGroup::draw_instanced_culled_rigged	(	Camera*					cam,
														unsigned int			frame,
														unsigned int			_AGENTS_NPOT,
														struct sVBOLod*			vbo_lods,
														GLuint&					posTextureTarget,
														GLuint&					posTextureId,
														float					dt,
														float*					viewMat,
														float*					projMat,
														float*					shadowMat,
														bool					wireframe,
														bool					shadows,
														bool					doHandD,
														bool					doPatterns,
														bool					doColor,
														bool					doFacial		)
{
	//err_manager->getError( "BEGIN : CharacterGroup::draw_instanced_culled_rigged" );
	for( unsigned int LOD = 0; LOD < NUM_LOD; LOD++ )
	{
		float dH = doHandD?1.0f:0.0f;
		float dP = doPatterns?1.0f:0.0f;
		float dC = doColor?1.0f:0.0f;
		float dF = doFacial?1.0f:0.0f;
		character_model[LOD]->draw_instanced_culled_rigged(	cam,
															frame,
															vbo_lods[LOD].primitivesWritten,
															_AGENTS_NPOT,
															animations[0]->getLength(),
															animations[0]->step,
															posTextureTarget,
															posTextureId,
															vbo_lods[LOD].id,
															clothingColorTableId,
															patternColorTableId,
															globalMtId,
															torsoMtId,
															legsMtId,
															riggingMtId,
															animations[0]->multitextureID,
															facialMtId,
															(float)LOD,		// Originalmente dt.
															(float)gender,
															viewMat,
															wireframe,
															dH,
															dP,
															dC,
															dF								);
		//printf( "[%d]: %03d\t", LOD, vbo_lods[LOD].primitivesWritten );
		if( shadows )
		{
			character_model[LOD_LO]->draw_instanced_culled_rigged_shadow(	cam,
																			frame,
																			vbo_lods[LOD].primitivesWritten,
																			_AGENTS_NPOT,
																			animations[0]->getLength(),
																			animations[0]->step,
																			posTextureTarget,
																			posTextureId,
																			vbo_lods[LOD].id,
																			riggingMtId,
																			animations[0]->multitextureID,
																			(float)LOD_LO,		// Originalmente dt.
																			(float)gender,
																			viewMat,
																			projMat,
																			shadowMat,
																			wireframe						);
		}
	}
	//printf( "\n" );
	//err_manager->getError( "END : CharacterGroup::draw_instanced_culled_rigged" );
}
#else
//
//=======================================================================================
//
void CharacterGroup::draw_instanced_culled_rigged	(	Camera*					cam,
														unsigned int			frame,
														unsigned int			_AGENTS_NPOT,
														struct sVBOLod*			vbo_lods,
														GLuint&					posTextureTarget,
														GLuint&					posTextureId,
														float					dt,
														float*					viewMat,
														float*					projMat,
														float*					shadowMat,
														bool					wireframe,
														bool					shadows			)
{
	//err_manager->getError( "BEGIN : CharacterGroup::draw_instanced_culled_rigged" );
	for( unsigned int LOD = 0; LOD < NUM_LOD; LOD++ )
	{
		character_model[LOD]->draw_instanced_culled_rigged(	cam,
															frame,
															vbo_lods[LOD].primitivesWritten,
															_AGENTS_NPOT,
															animations[0]->getLength(),
															animations[0]->step,
															posTextureTarget,
															posTextureId,
															vbo_lods[LOD].id,
															clothingColorTableId,
															patternColorTableId,
															globalMtId,
															torsoMtId,
															legsMtId,
															riggingMtId,
															animations[0]->multitextureID,
															facialMtId,
															(float)LOD,		// Originalmente dt.
															(float)gender,
															viewMat,
															wireframe						);
		//printf( "[%d]: %03d\t", LOD, vbo_lods[LOD].primitivesWritten );
		if( shadows )
		{
			character_model[LOD_LO]->draw_instanced_culled_rigged_shadow(	cam,
																			frame,
																			vbo_lods[LOD].primitivesWritten,
																			_AGENTS_NPOT,
																			animations[0]->getLength(),
																			animations[0]->step,
																			posTextureTarget,
																			posTextureId,
																			vbo_lods[LOD].id,
																			riggingMtId,
																			animations[0]->multitextureID,
																			(float)LOD_LO,		// Originalmente dt.
																			(float)gender,
																			viewMat,
																			projMat,
																			shadowMat,
																			wireframe						);
		}
	}
	//printf( "\n" );
	//err_manager->getError( "END : CharacterGroup::draw_instanced_culled_rigged" );
}
#endif
#endif
//
//=======================================================================================
