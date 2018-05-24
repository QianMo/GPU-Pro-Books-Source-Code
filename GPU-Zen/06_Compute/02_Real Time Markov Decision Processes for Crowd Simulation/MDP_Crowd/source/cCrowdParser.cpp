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
#include "cCrowdParser.h"

//=======================================================================================
//
CrowdParser::CrowdParser( LogManager* log_manager, char* fname )
{
	this->log_manager	= log_manager;
	filename			= fname;
	parser				= new XmlParser( this->log_manager, filename );
	NUM_CROWDS			= 0;
	NUM_GROUPS			= 0;
	NUM_MODEL_PROPS		= 0;
	rand_min			= 0;
	rand_max			= 0;
}
//
//=======================================================================================
//
CrowdParser::~CrowdParser( void )
{
	FREE_INSTANCE( parser );
	parsed_crowds.clear();
	parsed_groups.clear();
	parsed_model_props.clear();
	parsed_model_props_names_map.clear();
	parsed_groups_names_map.clear();
}
//
//=======================================================================================
//
bool CrowdParser::init( void )
{
	if( parser->init() )
	{
		parse();
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
void CrowdParser::parse( void )
{
    string empty( "" );
	unsigned int crowd			= 0;
	unsigned int model_props	= 0;
	unsigned int group			= 0;
	walk_tree( parser->getDoc(), crowd, model_props, group, (char*)empty.c_str() );

	for( unsigned int p = 0; p < parsed_model_props.size(); p++ )
	{
		string mpType = "";
		switch( parsed_model_props[p]->type )
		{
		case MPT_CLOTHING:
			mpType = "CLOTHING";
			break;
		case MPT_FACIAL:
			mpType = "FACIAL";
			break;
		case MPT_RIGGING:
			mpType = "RIGGING";
			break;
		}
		log_manager->file_log( LogManager::XML, "MODEL_PROPS::TYPE=\"%s\"::NAME=\"%s\"", mpType.c_str(), parsed_model_props[p]->name.c_str() );
		log_manager->file_log( LogManager::XML, "MODEL_PROPS::NUM_PROPS=%d", parsed_model_props[p]->parsed_props.size() );
	}

	for( unsigned int g = 0; g < parsed_groups.size(); g++ )
	{
		log_manager->file_log( LogManager::XML, "GROUP::NAME=\"%s\"", parsed_groups[g]->name.c_str() );
		log_manager->file_log( LogManager::XML, "GROUP::REF_CLOTHING_PROPS=\"%s\"", parsed_groups[g]->ref_clothing_props.c_str() );
	}

	for( unsigned int c = 0; c < parsed_crowds.size(); c++ )
	{
		log_manager->file_log( LogManager::XML, "CROWD::NAME=\"%s\"", parsed_crowds[c]->name.c_str() );
		log_manager->file_log( LogManager::XML, "CROWD::NUM_GROUPS=%d", parsed_crowds[c]->groups.size() );
	}
}
//
//=======================================================================================
//
void CrowdParser::walk_tree( TiXmlNode* pParent, unsigned int curr_crowd, unsigned int curr_model_props, unsigned int curr_group, char* curr_tag )
{
	if( !pParent )
	{
		return;
	}
	int t = pParent->Type();

	switch( t )
	{
		case TiXmlNode::TINYXML_ELEMENT:
			curr_tag = (char *)pParent->Value();
			if( strcmp( curr_tag, "model_props" ) == 0 )
			{
				ParsedModelProps *mProps = new ParsedModelProps();
				parsed_model_props.push_back( mProps );
				curr_model_props = parsed_model_props.size() - 1;

				TiXmlAttribute* pAttrib = pParent->ToElement()->FirstAttribute();
				if( pAttrib )
				{
					char* cType = (char*)pAttrib->Value();
					StringUtils::toUpper( cType );
					string sType = string( cType );
					if( sType.compare( "CLOTHING" ) == 0 )
					{
						parsed_model_props[curr_model_props]->type = MPT_CLOTHING;
					}
					else if( sType.compare( "FACIAL" ) == 0 )
					{
						parsed_model_props[curr_model_props]->type = MPT_FACIAL;
					}
					else if( sType.compare( "RIGGING" ) == 0 )
					{
						parsed_model_props[curr_model_props]->type = MPT_RIGGING;
					}
					else
					{
						log_manager->log( LogManager::WARNING, "\"%s\" Unknown MODEL_PROPS::TYPE", sType.c_str() );
					}
				}
				else
				{
					parsed_model_props[curr_model_props]->type = MPT_CLOTHING;
					log_manager->log( LogManager::LERROR, "\"%s\" Missing MODEL_PROPS::TYPE", filename );
				}

				pAttrib = pAttrib->Next();
				if( pAttrib )
				{
					char *name = (char *)pAttrib->Value();
					StringUtils::toUpper( name );
					parsed_model_props[curr_model_props]->name = string( name );
				}
				else
				{
					log_manager->log( LogManager::LERROR, "\"%s\" Missing MODEL_PROPS::NAME", filename );
				}

				pAttrib = pAttrib->Next();
				if( pAttrib )
				{
					char *cSex = (char *)pAttrib->Value();
					StringUtils::toUpper( cSex );
					string sex = string( cSex );
					if( sex.compare( "MALE" ) == 0 )
					{
						parsed_model_props[curr_model_props]->sex = MG_MALE;
					}
					else if( sex.compare( "FEMALE" ) == 0 )
					{
						parsed_model_props[curr_model_props]->sex = MG_FEMALE;
					}
				}
				else
				{
					log_manager->log( LogManager::LERROR, "\"%s\" Missing MODEL_PROPS::SEX", filename );
				}

				parsed_model_props_names_map[parsed_model_props[curr_model_props]->name] = NUM_MODEL_PROPS;
				NUM_MODEL_PROPS++;
			}
			else if( strcmp( curr_tag, "group" ) == 0 )
			{
				ParsedGroup *pGroup = new ParsedGroup();
				parsed_groups.push_back( pGroup );
				curr_group = parsed_groups.size() - 1;

				TiXmlAttribute* pAttrib = pParent->ToElement()->FirstAttribute();
				if( pAttrib )
				{
					char *name = (char *)pAttrib->Value();
					StringUtils::toUpper( name );
					parsed_groups[curr_group]->name = string( name );
				}
				else
				{
					log_manager->log( LogManager::LERROR, "\"%s\" Missing GROUP::NAME", filename );
				}

				pAttrib = pAttrib->Next();
				if( pAttrib )
				{
					char *cClothingProps = (char *)pAttrib->Value();
					StringUtils::toUpper( cClothingProps );
					parsed_groups[curr_group]->ref_clothing_props = string( cClothingProps );
				}
				else
				{
					log_manager->log( LogManager::LERROR, "\"%s\" Missing GROUP::REF_CLOTHING_PROPS", filename );
				}

				pAttrib = pAttrib->Next();
				if( pAttrib )
				{
					char *cFacialProps = (char *)pAttrib->Value();
					StringUtils::toUpper( cFacialProps );
					parsed_groups[curr_group]->ref_facial_props = string( cFacialProps );
				}
				else
				{
					log_manager->log( LogManager::LERROR, "\"%s\" Missing GROUP::REF_FACIAL_PROPS", filename );
				}

				pAttrib = pAttrib->Next();
				if( pAttrib )
				{
					char *cRiggingProps = (char *)pAttrib->Value();
					StringUtils::toUpper( cRiggingProps );
					parsed_groups[curr_group]->ref_rigging_props = string( cRiggingProps );
				}
				else
				{
					log_manager->log( LogManager::LERROR, "\"%s\" Missing GROUP::REF_RIGGING_PROPS", filename );
				}

				pAttrib = pAttrib->Next();
				if( pAttrib )
				{
					char *cSex = (char *)pAttrib->Value();
					StringUtils::toUpper( cSex );
					string sSex = string( cSex );
					if( sSex.compare( "MALE" ) == 0 )
					{
						parsed_groups[curr_group]->sex = MG_MALE;
					}
					else if( sSex.compare( "FEMALE" ) == 0 )
					{
						parsed_groups[curr_group]->sex = MG_FEMALE;
					}
					else
					{
						log_manager->log( LogManager::WARNING, "\"%s\" Unknown gender.", sSex.c_str() );
					}
				}
				else
				{
					log_manager->log( LogManager::LERROR, "\"%s\" Missing GROUP::SEX", filename );
				}

				pAttrib = pAttrib->Next();
				if( pAttrib )
				{
					char *cType = (char *)pAttrib->Value();
					StringUtils::toUpper( cType );
					string sType = string( cType );
					if( sType.compare( "HUMAN" ) == 0 )
					{
						parsed_groups[curr_group]->type = MT_HUMAN;
					}
					else if( sType.compare( "LEMMING" ) == 0 )
					{
						parsed_groups[curr_group]->type = MT_LEMMING;
					}
					else
					{
						log_manager->log( LogManager::WARNING, "\"%s\" Unknown type.", sType.c_str() );
					}
				}
				else
				{
					log_manager->log( LogManager::LERROR, "\"%s\" Missing GROUP::TYPE", filename );
				}


				parsed_groups_names_map[parsed_groups[curr_group]->name] = NUM_GROUPS;
				NUM_GROUPS++;
			}
			else if( strcmp( curr_tag, "crowd" ) == 0 )
			{
				ParsedCrowd *pCrowd = new ParsedCrowd();
				parsed_crowds.push_back( pCrowd );
				curr_crowd = parsed_crowds.size() - 1;

				TiXmlAttribute* pAttrib = pParent->ToElement()->FirstAttribute();
				if( pAttrib )
				{
					char *name = (char *)pAttrib->Value();
					StringUtils::toUpper( name );
					parsed_crowds[curr_crowd]->name = string( name );
				}
				else
				{
					log_manager->log( LogManager::LERROR, "\"%s\" Missing CROWD::NAME", filename );
				}

				pAttrib = pAttrib->Next();
				if( pAttrib )
				{
					unsigned int WIDTH = 0;
					char *cWIDTH = (char *)pAttrib->Value();
					std::stringstream ss1( cWIDTH );
					ss1 >> WIDTH;
					parsed_crowds[curr_crowd]->width = WIDTH;
				}
				else
				{
					log_manager->log( LogManager::LERROR, "\"%s\" Missing CROWD::WIDTH", filename );
				}

				pAttrib = pAttrib->Next();
				if( pAttrib )
				{
					unsigned int HEIGHT = 0;
					char *cHEIGHT = (char *)pAttrib->Value();
					std::stringstream ss1( cHEIGHT );
					ss1 >> HEIGHT;
					parsed_crowds[curr_crowd]->height = HEIGHT;
				}
				else
				{
					log_manager->log( LogManager::LERROR, "\"%s\" Missing CROWD::HEIGHT", filename );
				}

				NUM_CROWDS++;
			}
			else if( strcmp( curr_tag, "prop" ) == 0 )
			{
				TiXmlAttribute* pAttrib = pParent->ToElement()->FirstAttribute();
				char* cType = (char*)pAttrib->Value();
				StringUtils::toUpper( cType );
				string sType = string( cType );

				pAttrib = pAttrib->Next();
				char* cSubtype = (char*)pAttrib->Value();
				StringUtils::toUpper( cSubtype );
				string sSubtype = string( cSubtype );

				pAttrib = pAttrib->Next();
				char* cName = (char*)pAttrib->Value();
				StringUtils::toUpper( cName );

				pAttrib = pAttrib->Next();
				char *cFile = (char*)pAttrib->Value();

				ParsedProp* pProp = new ParsedProp();

				if( sType.compare( "ATLAS" ) == 0 )
				{
					pProp->type = PP_ATLAS;
				}
				else if( sType.compare( "WRINKLES" ) == 0 )
				{
					pProp->type = PP_WRINKLES;
				}
				else if( sType.compare( "PATTERN" ) == 0 )
				{
					pProp->type = PP_PATTERN;
				}
				else if( sType.compare( "HEAD" ) == 0 )
				{
					pProp->type = PP_HEAD;
				}
				else if( sType.compare( "HAIR" ) == 0 )
				{
					pProp->type = PP_HAIR;
				}
				else if( sType.compare( "TORSO" ) == 0 )
				{
					pProp->type = PP_TORSO;
				}
				else if( sType.compare( "LEGS" ) == 0 )
				{
					pProp->type = PP_LEGS;
				}
				else if( sType.compare( "FACIAL_WRINKLES" ) == 0 )
				{
					pProp->type = PP_FACIAL_WRINKLES;
				}
				else if( sType.compare( "FACIAL_EYE_SOCKETS" ) == 0 )
				{
					pProp->type = PP_FACIAL_EYE_SOCKETS;
				}
				else if( sType.compare( "FACIAL_SPOTS" ) == 0 )
				{
					pProp->type = PP_FACIAL_SPOTS;
				}
				else if( sType.compare( "FACIAL_BEARD" ) == 0 )
				{
					pProp->type = PP_FACIAL_BEARD;
				}
				else if( sType.compare( "FACIAL_MOUSTACHE" ) == 0 )
				{
					pProp->type = PP_FACIAL_MOUSTACHE;
				}
				else if( sType.compare( "FACIAL_MAKEUP" ) == 0 )
				{
					pProp->type = PP_FACIAL_MAKEUP;
				}
				else if( sType.compare( "RIGGING_ZONES" ) == 0 )
				{
					pProp->type = PP_RIGGING_ZONES;
				}
				else if( sType.compare( "RIGGING_WEIGHTS" ) == 0 )
				{
					pProp->type = PP_RIGGING_WEIGHTS;
				}
				else if( sType.compare( "RIGGING_DISPLACEMENT" ) == 0 )
				{
					pProp->type = PP_RIGGING_DISPLACEMENT;
				}
				else if( sType.compare( "RIGGING_ANIMATION" ) == 0 )
				{
					pProp->type = PP_RIGGING_ANIMATION;
				}
				else
				{
					log_manager->log( LogManager::LERROR, "\"%s\". Unknown Type.", sType.c_str() );
				}

				if( sSubtype.compare( "SKIN" ) == 0 )
				{
					pProp->subtype = PST_SKIN;
				}
				else if( sSubtype.compare( "HEAD" ) == 0 )
				{
					pProp->subtype = PST_HEAD;
				}
				else if( sSubtype.compare( "HAIR" ) == 0 )
				{
					pProp->subtype = PST_HAIR;
				}
				else if( sSubtype.compare( "CAP" ) == 0 )
				{
					pProp->subtype = PST_CAP;
				}
				else if( sSubtype.compare( "TORSO" ) == 0 )
				{
					pProp->subtype = PST_TORSO;
				}
				else if( sSubtype.compare( "LEGS" ) == 0 )
				{
					pProp->subtype = PST_LEGS;
				}
				else if( sSubtype.compare( "TORSO_AND_LEGS" ) == 0 )
				{
					pProp->subtype = PST_TORSO_AND_LEGS;
				}
				else if( sSubtype.compare( "FACE" ) == 0 )
				{
					pProp->subtype = PST_FACE;
				}
				else if( sSubtype.compare( "RIG" ) == 0 )
				{
					pProp->subtype = PST_RIG;
				}
				else
				{
					log_manager->log( LogManager::LERROR, "\"%s\". Unknown Subtype.", sSubtype.c_str() );
				}

				pProp->name = string( cName );
				pProp->file = string( cFile );

				parsed_model_props[curr_model_props]->props_names_map[pProp->name] = parsed_model_props[curr_model_props]->parsed_props.size();
				parsed_model_props[curr_model_props]->parsed_props.push_back( pProp );
			}
			else if( strcmp( curr_tag, "wrinkles" ) == 0 )
			{
				TiXmlAttribute* pAttrib = pParent->ToElement()->FirstAttribute();
				char* cRef = (char*)pAttrib->Value();
				StringUtils::toUpper( cRef );

				unsigned int last = parsed_model_props[curr_model_props]->parsed_props.size() - 1;
				parsed_model_props[curr_model_props]->parsed_props[last]->ref_wrinkles = string( cRef );
			}
			else if( strcmp( curr_tag, "pattern" ) == 0 )
			{
				TiXmlAttribute* pAttrib = pParent->ToElement()->FirstAttribute();
				char* cRef = (char*)pAttrib->Value();
				StringUtils::toUpper( cRef );

				unsigned int last = parsed_model_props[curr_model_props]->parsed_props.size() - 1;
				parsed_model_props[curr_model_props]->parsed_props[last]->ref_pattern.push_back( string( cRef ) );
			}
			else if( strcmp( curr_tag, "atlas" ) == 0 )
			{
				TiXmlAttribute* pAttrib = pParent->ToElement()->FirstAttribute();
				char* cRef = (char*)pAttrib->Value();
				StringUtils::toUpper( cRef );

				unsigned int last = parsed_model_props[curr_model_props]->parsed_props.size() - 1;
				parsed_model_props[curr_model_props]->parsed_props[last]->ref_atlas.push_back( string( cRef ) );
			}
			else if( strcmp( curr_tag, "outfit" ) == 0 )
			{
				TiXmlAttribute* pAttrib = pParent->ToElement()->FirstAttribute();
				char *cHead = (char *)pAttrib->Value();
				StringUtils::toUpper( cHead );

				pAttrib = pAttrib->Next();
				char *cHair = (char *)pAttrib->Value();
				StringUtils::toUpper( cHair );

				pAttrib = pAttrib->Next();
				char *cLegs = (char *)pAttrib->Value();
				StringUtils::toUpper( cLegs );

				pAttrib = pAttrib->Next();
				char *cTorso = (char *)pAttrib->Value();
				StringUtils::toUpper( cTorso );

				pAttrib = pAttrib->Next();
				char *cPalette = (char *)pAttrib->Value();
				StringUtils::toUpper( cPalette );

				parsed_groups[curr_group]->ref_head		= string( cHead		);
				parsed_groups[curr_group]->ref_hair		= string( cHair		);
				parsed_groups[curr_group]->ref_legs		= string( cLegs		);
				parsed_groups[curr_group]->ref_torso	= string( cTorso	);
				parsed_groups[curr_group]->ref_palette	= string( cPalette	);
			}
			else if( strcmp( curr_tag, "face" ) == 0 )
			{
				TiXmlAttribute* pAttrib = pParent->ToElement()->FirstAttribute();
				char *cWrinkles = (char *)pAttrib->Value();
				StringUtils::toUpper( cWrinkles );

				pAttrib = pAttrib->Next();
				char *cEye_sockets = (char *)pAttrib->Value();
				StringUtils::toUpper( cEye_sockets );

				pAttrib = pAttrib->Next();
				char *cSpots = (char *)pAttrib->Value();
				StringUtils::toUpper( cSpots );

				pAttrib = pAttrib->Next();
				char *cBeard = (char *)pAttrib->Value();
				StringUtils::toUpper( cBeard );

				pAttrib = pAttrib->Next();
				char *cMoustache = (char *)pAttrib->Value();
				StringUtils::toUpper( cMoustache );

				pAttrib = pAttrib->Next();
				char *cMakeup = (char *)pAttrib->Value();
				StringUtils::toUpper( cMakeup );

				parsed_groups[curr_group]->ref_wrinkles		= string( cWrinkles		);
				parsed_groups[curr_group]->ref_eye_sockets	= string( cEye_sockets	);
				parsed_groups[curr_group]->ref_spots		= string( cSpots		);
				parsed_groups[curr_group]->ref_beard		= string( cBeard		);
				parsed_groups[curr_group]->ref_moustache	= string( cMoustache	);
				parsed_groups[curr_group]->ref_makeup		= string( cMakeup		);
			}
			else if( strcmp( curr_tag, "rig" ) == 0 )
			{
				TiXmlAttribute* pAttrib = pParent->ToElement()->FirstAttribute();
				char *cZones = (char *)pAttrib->Value();
				StringUtils::toUpper( cZones );

				pAttrib = pAttrib->Next();
				char *cWeights = (char *)pAttrib->Value();
				StringUtils::toUpper( cWeights );

				pAttrib = pAttrib->Next();
				char *cDisplacement = (char *)pAttrib->Value();
				StringUtils::toUpper( cDisplacement );

				pAttrib = pAttrib->Next();
				char *cAnimation = (char *)pAttrib->Value();
				StringUtils::toUpper( cAnimation );

				parsed_groups[curr_group]->ref_zones		= string( cZones		);
				parsed_groups[curr_group]->ref_weights		= string( cWeights		);
				parsed_groups[curr_group]->ref_displacement	= string( cDisplacement	);
				parsed_groups[curr_group]->ref_animation	= string( cAnimation	);
			}
			else if( strcmp( curr_tag, "weight_size" ) == 0 )
			{
				TiXmlAttribute* pAttrib = pParent->ToElement()->FirstAttribute();
				char *cFAT = (char *)pAttrib->Value();
				float FAT;
				std::istringstream iss1( cFAT );
				iss1 >> FAT;

				pAttrib = pAttrib->Next();
				char *cAVERAGE = (char *)pAttrib->Value();
				float AVERAGE;
				std::istringstream iss2( cAVERAGE );
				iss2 >> AVERAGE;

				pAttrib = pAttrib->Next();
				char *cTHIN = (char *)pAttrib->Value();
				float THIN;
				std::istringstream iss3( cTHIN );
				iss3 >> THIN;

				pAttrib = pAttrib->Next();
				char *cSTRONG = (char *)pAttrib->Value();
				float STRONG;
				std::istringstream iss4( cSTRONG );
				iss4 >> STRONG;

				float sum = FAT + AVERAGE + THIN + STRONG;
				if( sum > 1.0f )
				{
					log_manager->log( LogManager::WARNING, "GROUP \"%s\" Weights sum > 1.0. Correcting AVERAGE.", parsed_groups[curr_group]->name.c_str() );
					AVERAGE = 1.0f - (FAT + THIN + STRONG);
				}
				else if( sum < 1.0f )
				{
					log_manager->log( LogManager::WARNING, "GROUP \"%s\" Weights sum < 1.0. Correcting AVERAGE.", parsed_groups[curr_group]->name.c_str() );
					AVERAGE = 1.0f - (FAT + THIN + STRONG);
				}

				parsed_groups[curr_group]->fat		= FAT;
				parsed_groups[curr_group]->average	= AVERAGE;
				parsed_groups[curr_group]->thin		= THIN;
				parsed_groups[curr_group]->strong	= STRONG;
			}
			else if( strcmp( curr_tag, "height_size" ) == 0 )
			{
				TiXmlAttribute* pAttrib = pParent->ToElement()->FirstAttribute();
				char *cMIN = (char *)pAttrib->Value();
				float MIN;
				std::istringstream iss1( cMIN );
				iss1 >> MIN;

				pAttrib = pAttrib->Next();
				char *cMAX = (char *)pAttrib->Value();
				float MAX;
				std::istringstream iss2( cMAX );
				iss2 >> MAX;

				parsed_groups[curr_group]->min = MIN;
				parsed_groups[curr_group]->max = MAX;
			}
			else if( strcmp( curr_tag, "add_group" ) == 0 )
			{
				TiXmlAttribute* pAttrib = pParent->ToElement()->FirstAttribute();
				char *cRef = (char *)pAttrib->Value();
				StringUtils::toUpper( cRef );

				pAttrib = pAttrib->Next();
				char *cPERCENTAGE = (char *)pAttrib->Value();
				float PERCENTAGE;
				std::istringstream iss1( cPERCENTAGE );
				iss1 >> PERCENTAGE;

				pAttrib = pAttrib->Next();
				char *cAnimation = (char *)pAttrib->Value();
				StringUtils::toUpper( cAnimation );

				pAttrib = pAttrib->Next();
				unsigned int ANIMATION_FRAMES = 0;
				char *cANIMATION_FRAMES = (char *)pAttrib->Value();
				std::stringstream ss2( cANIMATION_FRAMES );
				ss2 >> ANIMATION_FRAMES;

				pAttrib = pAttrib->Next();
				unsigned int ANIMATION_DURATION = 0;
				char *cANIMATION_DURATION = (char *)pAttrib->Value();
				std::stringstream ss3( cANIMATION_DURATION );
				ss3 >> ANIMATION_DURATION;

				ParsedCrowdGroup* pcg	= new ParsedCrowdGroup();
				pcg->ref				= string( cRef );
				pcg->percentage			= PERCENTAGE;
				pcg->animation_frames	= ANIMATION_FRAMES;
				pcg->animation_duration	= ANIMATION_DURATION;
				pcg->animation			= string( cAnimation );

				parsed_crowds[curr_crowd]->groups.push_back( pcg );
			}
			break;
		case TiXmlNode::TINYXML_TEXT:
			break;
		default:
			break;
	}

	TiXmlNode *pChild = pParent->FirstChild();
	while( pChild )
	{
		walk_tree( pChild, curr_crowd, curr_model_props, curr_group, curr_tag );
		pChild = pChild->NextSibling();
	}
}
//
//=======================================================================================
//
vector<ParsedCrowd *>& CrowdParser::getParsed_Crowds( void )
{
	return parsed_crowds;
}
//
//=======================================================================================
//
vector<ParsedGroup *>& CrowdParser::getParsed_Groups( void )
{
	return parsed_groups;
}
//
//=======================================================================================
//
vector<ParsedModelProps *>& CrowdParser::getParsed_ModelProps( void )
{
	return parsed_model_props;
}
//
//=======================================================================================
//
unsigned int CrowdParser::getNumCrowds( void )
{
	return NUM_CROWDS;
}
//
//=======================================================================================
//
unsigned int CrowdParser::getNumGroups( void )
{
	return NUM_GROUPS;
}
//
//=======================================================================================
//
unsigned int CrowdParser::getNumModelProps( void )
{
	return NUM_MODEL_PROPS;
}
//
//=======================================================================================
//
float CrowdParser::getRandom( void )
{
	int sign = 1;
	int s = 1 + (rand() % 2);
	if( s == 2 )
	{
		sign = -1;
	}
	int mod = rand_max - rand_min;
	float r = (float)(rand_min + (rand() % mod));
	r *= sign;
	return r;
}
//
//=======================================================================================
