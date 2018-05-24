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

//=======================================================================================
//
void init_globals( void )
{
	//scenario_type				= ST_NONE;
	//scenario_type				= ST_O1P2;
	//scenario_type				= ST_EIFFEL;
	//scenario_type				= ST_TOWN;
	//scenario_type				= ST_MAZE;
	//scenario_type				= ST_CCM;
	
#if defined MDPS_SQUARE_NOLCA
	if( MDP_CHANNELS == 1 )
	{
		string mdp_csv			= string( "assets/mdp/OPEN/mdp_001_10x10.csv"			);
		str_mdp_csv.push_back( mdp_csv );
	}
	else if( MDP_CHANNELS == 2 )
	{
		string mdp_csv			= string( "assets/mdp/OPEN/mdp_001_10x10.csv"			);
		str_mdp_csv.push_back( mdp_csv );
		str_mdp_csv.push_back( mdp_csv );
	}
	str_scenario_speed			= string( "assets/mdp/OPEN/speed_001_10x10.csv"			);
	//obstacle_type				= OT_NONE;
	//obstacle_type				= OT_BARREL;
	obstacle_type				= OT_BOX;
	crowd_position				= CPOS_NONE;
#elif defined MDPS_SQUARE_LCA
	if( scenario_type == ST_NONE )
	{
		if( MDP_CHANNELS == 1 )
		{
			string mdp_csv		= string( "assets/mdp/OPEN/mdp_001_5x5.csv"				);
			//string mdp_csv	= string( "assets/mdp/OPEN/mdp_001_10x10.csv"			);
			str_mdp_csv.push_back( mdp_csv );
			str_scenario_speed	= string( "assets/mdp/OPEN/speed_001_5x5.csv"			);
		}
		else if( MDP_CHANNELS == 2 )
		{
			//string mdp_csv1	= string( "assets/mdp/OPEN/mdp_bottom_5x5.csv"			);
			//string mdp_csv2	= string( "assets/mdp/OPEN/mdp_top_5x5.csv"				);
			string mdp_csv1		= string( "assets/mdp/CROSS/cross_left_10x10.csv"		);
			string mdp_csv2		= string( "assets/mdp/CROSS/cross_down_10x10.csv"		);
			str_mdp_csv.push_back( mdp_csv1 );
			str_mdp_csv.push_back( mdp_csv2 );
			str_scenario_speed	= string( "assets/mdp/CROSS/speed_cross_10x10.csv"		);
		}
		obstacle_type			= OT_GLASSBOX;
		agent_scale				= 1.0f;
		crowd_position			= CPOS_NONE;

		str_skybox_front		= ( "assets/skybox/rays_north.dds"						);
		str_skybox_back			= ( "assets/skybox/rays_south.dds"						);
		str_skybox_left			= ( "assets/skybox/rays_west.dds"						);
		str_skybox_right		= ( "assets/skybox/rays_east.dds"						);
		str_skybox_top			= ( "assets/skybox/rays_up.dds"							);
		str_skybox_bottom		= ( "assets/skybox/rays_down.dds"						);
	}
	else if( scenario_type == ST_O1P2 )
	{
		if( MDP_CHANNELS == 1 )
		{
			string mdp_csv		= string( "assets/mdp/O1P2/o1p2_60x60.csv"				);
			//string mdp_csv		= string( "assets/mdp/O1P2/o1p2_38x38.csv"				);
			str_mdp_csv.push_back( mdp_csv );
		}
		else if( MDP_CHANNELS == 2 )
		{
			string mdp_csv1		= string( "assets/mdp/O1P2/o1p2_right_60x60.csv"		);
			string mdp_csv2		= string( "assets/mdp/O1P2/o1p2_left_60x60.csv"			);
			//string mdp_csv1		= string( "assets/mdp/O1P2/o1p2_right_38x38.csv"		);
			//string mdp_csv2		= string( "assets/mdp/O1P2/o1p2_left_38x38.csv"			);
			str_mdp_csv.push_back( mdp_csv1 );
			str_mdp_csv.push_back( mdp_csv2 );
		}
		str_scenario_speed		= string( "assets/mdp/O1P2/speed_o1p2_60x60.csv"		);
		//str_scenario_speed		= string( "assets/mdp/O1P2/speed_o1p2_38x38.csv"		);
		agent_scale				= 7.0f;
		obstacle_type			= OT_NONE;
		//crowd_position		= CPOS_NONE;
		crowd_position			= CPOS_TOP;

		str_skybox_front		= ( "assets/skybox/ccm_north.dds"						);
		str_skybox_back			= ( "assets/skybox/ccm_south.dds"						);
		str_skybox_left			= ( "assets/skybox/ccm_west.dds"						);
		str_skybox_right		= ( "assets/skybox/ccm_east.dds"						);
		str_skybox_top			= ( "assets/skybox/ccm_up.dds"							);
		str_skybox_bottom		= ( "assets/skybox/ccm_down.dds"						);
	}
	else if( scenario_type == ST_EIFFEL )
	{
		if( MDP_CHANNELS == 1 )
		{
			string mdp_csv		= string( "assets/mdp/EIFFEL/eiffel_100x100.csv"		);

			str_mdp_csv.push_back( mdp_csv );
		}
		else if( MDP_CHANNELS == 2 )
		{
			string mdp_csv		= string( "assets/mdp/EIFFEL/eiffel_100x100.csv"		);
			str_mdp_csv.push_back( mdp_csv );
			str_mdp_csv.push_back( mdp_csv );
		}
		str_scenario_speed		= string( "assets/mdp/EIFFEL/speed_eiffel_100x100.csv"	);
		agent_scale				= 0.3f;
		crowd_position			= CPOS_NONE;

#ifdef DRAW_OBSTACLES
		obstacle_type			= OT_BOX;
#else
		obstacle_type			= OT_NONE;
#endif

		str_skybox_front		= ( "assets/skybox/eiffel_north.dds"					);
		str_skybox_back			= ( "assets/skybox/eiffel_south.dds"					);
		str_skybox_left			= ( "assets/skybox/eiffel_west.dds"						);
		str_skybox_right		= ( "assets/skybox/eiffel_east.dds"						);
		str_skybox_top			= ( "assets/skybox/eiffel_up.dds"						);
		str_skybox_bottom		= ( "assets/skybox/eiffel_down.dds"						);
	}
	else if( scenario_type == ST_TOWN )
	{
		if( MDP_CHANNELS == 1 )
		{
			//string mdp_csv	= string( "assets/mdp/TOWN/town_100x100.csv"			);
			string mdp_csv		= string( "assets/mdp/TOWN/town_to_houses_100x100.csv"	);
			str_mdp_csv.push_back( mdp_csv );
		}
		else if( MDP_CHANNELS == 2 )
		{
			string mdp_csv		= string( "assets/mdp/TOWN/town_to_houses_100x100.csv"	);
			str_mdp_csv.push_back( mdp_csv );
			str_mdp_csv.push_back( mdp_csv );
		}
		str_scenario_speed		= string( "assets/mdp/TOWN/speed_town_100x100.csv"		);
		agent_scale				= 1.0f;
		obstacle_type			= OT_NONE;
		crowd_position			= CPOS_NONE;

		str_skybox_front		= ( "assets/skybox/canyon_north.dds"					);
		str_skybox_back			= ( "assets/skybox/canyon_south.dds"					);
		str_skybox_left			= ( "assets/skybox/canyon_west.dds"						);
		str_skybox_right		= ( "assets/skybox/canyon_east.dds"						);
		str_skybox_top			= ( "assets/skybox/canyon_up.dds"						);
		str_skybox_bottom		= ( "assets/skybox/canyon_down.dds"						);
	}
	else if( scenario_type == ST_MAZE )
	{
		if( MDP_CHANNELS == 1 )
		{
			string mdp_csv		= string( "assets/mdp/MAZE/maze_100x100.csv"			);
			str_mdp_csv.push_back( mdp_csv );
		}
		else if( MDP_CHANNELS == 2 )
		{
			string mdp_csv		= string( "assets/mdp/MAZE/maze_100x100.csv"			);
			str_mdp_csv.push_back( mdp_csv );
			str_mdp_csv.push_back( mdp_csv );
		}
		str_scenario_speed		= string( "assets/mdp/MAZE/speed_maze_100x100.csv"		);
		agent_scale				= 1.0f;
		obstacle_type			= OT_GLASSBOX;
		crowd_position			= CPOS_NONE;

		str_skybox_front		= ( "assets/skybox/lagoon_north.dds"					);
		str_skybox_back			= ( "assets/skybox/lagoon_south.dds"					);
		str_skybox_left			= ( "assets/skybox/lagoon_west.dds"						);
		str_skybox_right		= ( "assets/skybox/lagoon_east.dds"						);
		str_skybox_top			= ( "assets/skybox/lagoon_up.dds"						);
		str_skybox_bottom		= ( "assets/skybox/lagoon_down.dds"						);
	}
	else if( scenario_type == ST_CCM )
	{
		if( MDP_CHANNELS == 1 )
		{
			string mdp_csv		= string( "assets/mdp/CCM/ccm_200x200.csv"				);
			str_mdp_csv.push_back( mdp_csv );
		}
		else if( MDP_CHANNELS == 2 )
		{
			string mdp_csv		= string( "assets/mdp/CCM/ccm_200x200.csv"				);
			str_mdp_csv.push_back( mdp_csv );
			str_mdp_csv.push_back( mdp_csv );
		}
		str_scenario_speed		= string( "assets/mdp/CCM/speed_ccm_200x200.csv"		);
		agent_scale				= 0.4f;
		//obstacle_type			= OT_NONE;
		//crowd_position			= CPOS_NONE;

		str_skybox_front		= ( "assets/skybox/rays_north.dds"						);
		str_skybox_back			= ( "assets/skybox/rays_south.dds"						);
		str_skybox_left			= ( "assets/skybox/rays_west.dds"						);
		str_skybox_right		= ( "assets/skybox/rays_east.dds"						);
		str_skybox_top			= ( "assets/skybox/rays_up.dds"							);
		str_skybox_bottom		= ( "assets/skybox/rays_down.dds"						);
	}

#elif defined MDPS_HEXAGON_NOLCA || defined MDPS_HEXAGON_LCA
	if( MDP_CHANNELS == 1 )
	{
		//string mdp_csv		= string( "assets/mdp/HEX/mdp_001_5x5h.csv"				);
		string mdp_csv			= string( "assets/mdp/HEX/mdp_001_10x10h.csv"			);
		//string mdp_csv		= string( "assets/mdp/HEX/mdp_001_20x20h.csv"			);
		//string mdp_csv		= string( "assets/mdp/HEX/mdp_002_20x20h.csv"			);
		//string mdp_csv		= string( "assets/mdp/HEX/mdp_002_10x10h.csv"			);
		//string mdp_csv		= string( "assets/mdp/HEX/mdp_003_10x10h.csv"			);
		str_mdp_csv.push_back( mdp_csv );
	}
	else if( MDP_CHANNELS == 2 )
	{
		string mdp_csv = string( "assets/mdp/HEX/mdp_001_10x10h.csv"					);
		str_mdp_csv.push_back( mdp_csv );
		str_mdp_csv.push_back( mdp_csv );
	}
	str_scenario_speed			= string( "assets/mdp/HEX/speed_001_10x10.csv"			);
	crowd_position				= CPOS_NONE;
#endif

	log_manager->log( LogManager::INFORMATION, "Global variables initialization complete." );
}
//
//=======================================================================================
//
void init_glsl( void )
{
	glsl_manager = new GlslManager( err_manager, log_manager );
	vector<InputShader*> input_shaders;

	string sName;
	string sVert;
	string sFrag;
	string sGeom;

	sName = string( "lambert" );
	sVert = string( "shaders/lambert.vert" );
	sFrag = string( "shaders/lambert.frag" );
	InputShader* input_lambert = new InputShader( sName );
	input_lambert->s_vert = sVert;
	input_lambert->s_frag = sFrag;
	input_lambert->s_uni_i["diffuseTexture"] = 0;
    input_lambert->s_uni_i["textured"] = 0;
	input_lambert->s_uni_f["opacity"] = 1.0f;
	InputShader::FV fv_tint;
	fv_tint.push_back( 0.5f );
	fv_tint.push_back( 0.5f );
	fv_tint.push_back( 0.5f );
	input_lambert->s_uni_fv["tint"] = fv_tint;
	InputShader::FV fv_amb;
	fv_amb.push_back( 0.12f );
	fv_amb.push_back( 0.12f );
	fv_amb.push_back( 0.12f );
	input_lambert->s_uni_fv["amb"] = fv_amb;
	input_shaders.push_back( input_lambert );

	sName = string( "gouraud" );
	sVert = string( "shaders/gouraud.vert" );
	sFrag = string( "shaders/gouraud.frag" );
	InputShader* input_gouraud = new InputShader( sName );
	input_gouraud->s_vert = sVert;
	input_gouraud->s_frag = sFrag;
	input_gouraud->s_uni_i["diffuseTexture"] = 0;
	input_shaders.push_back( input_gouraud );

#if defined MDPS_SQUARE_NOLCA || defined MDPS_SQUARE_LCA
	sName = string( "mdp_floor" );
	sVert = string( "shaders/mdp_floor_vertex.glsl" );
	sFrag = string( "shaders/mdp_floor_fragment.glsl" );
	InputShader* input_mdp_floor = new InputShader( sName );
	input_mdp_floor->s_vert = sVert;
	input_mdp_floor->s_frag = sFrag;
	input_mdp_floor->s_uni_i["policy"]			= 0;
	input_mdp_floor->s_uni_i["density"]			= 1;
	input_mdp_floor->s_uni_i["arrowsTA"]		= 2;
	input_mdp_floor->s_uni_i["layer0"]			= 3;
	input_mdp_floor->s_uni_i["layer1"]			= 4;
	input_mdp_floor->s_uni_f["tile_width"]		= 100.0f;
	input_mdp_floor->s_uni_f["tile_height"]		= 100.0f;
	input_mdp_floor->s_uni_f["fbo_width"]		= 2000.0f;
	input_mdp_floor->s_uni_f["fbo_height"]		= 2000.0f;
	input_mdp_floor->s_uni_f["policy_width"]	= 10.0f;
	input_mdp_floor->s_uni_f["policy_height"]	= 10.0f;
	input_shaders.push_back( input_mdp_floor );
#elif defined MDPS_HEXAGON_NOLCA || defined MDPS_HEXAGON_LCA
	sName = string( "mdp_floor" );
	sVert = string( "shaders/mdp_floor_hex_vertex.glsl" );
	sFrag = string( "shaders/mdp_floor_hex_fragment.glsl" );
	InputShader* input_mdp_floor				= new InputShader( sName );
	input_mdp_floor->s_vert						= sVert;
	input_mdp_floor->s_frag						= sFrag;
	input_mdp_floor->s_uni_i["policy"]			= 0;
	input_mdp_floor->s_uni_i["density"]			= 1;
	input_mdp_floor->s_uni_i["arrowsTA"]		= 2;
	input_mdp_floor->s_uni_i["layer0"]			= 3;
	input_mdp_floor->s_uni_i["layer1"]			= 4;
	//input_mdp_floor->s_uni_f["tile_width"]	= 115.470053838f;	// W = 2*R			R =57.7350269f
	input_mdp_floor->s_uni_f["tile_side"]		= 86.6025404f;		// S = (3/2)*R
	input_mdp_floor->s_uni_f["tile_height"]		= 100.0f;			// H = SQRT(3)*R;	H = 2*R*sin(60)
	input_mdp_floor->s_uni_f["fbo_width"]		= 1000.0f;
	input_mdp_floor->s_uni_f["fbo_height"]		= 1000.0f;
	input_mdp_floor->s_uni_f["policy_width"]	= 10.0f;
	input_mdp_floor->s_uni_f["policy_height"]	= 10.0f;
	input_shaders.push_back( input_mdp_floor );
#endif

	sName = string( "instancing_textured" );
	sVert = string( "shaders/instancing_textured.vert" );
	sFrag = string( "shaders/instancing_textured.frag" );
	InputShader* input_instancing_textured = new InputShader( sName );
	input_instancing_textured->s_vert = sVert;
	input_instancing_textured->s_frag = sFrag;
	input_instancing_textured->s_uni_i["diffuseTexture"] = 0;
	input_instancing_textured->s_uni_f["opacity"] = 1.0f;
	InputShader::FV fv_tinti;
	fv_tinti.push_back( 0.5f );
	fv_tinti.push_back( 0.5f );
	fv_tinti.push_back( 0.5f );
	input_instancing_textured->s_uni_fv["tint"] = fv_tinti;
	InputShader::FV fv_ambi;
	fv_ambi.push_back( 0.12f );
	fv_ambi.push_back( 0.12f );
	fv_ambi.push_back( 0.12f );
	input_instancing_textured->s_uni_fv["amb"] = fv_ambi;
	input_shaders.push_back( input_instancing_textured );

	sName = string( "instancing_untextured" );
	sVert = string( "shaders/instancing_untextured.vert" );
	sFrag = string( "shaders/instancing_untextured.frag" );
	InputShader* input_instancing_untextured = new InputShader( sName );
	input_instancing_untextured->s_vert = sVert;
	input_instancing_untextured->s_frag = sFrag;
	InputShader::FV fv_ambii;
	fv_ambii.push_back( 0.12f );
	fv_ambii.push_back( 0.12f );
	fv_ambii.push_back( 0.12f );
	input_instancing_untextured->s_uni_fv["amb"] = fv_ambii;
	input_instancing_untextured->s_uni_f["opacity"] = 1.0f;
	InputShader::FV fv_tintii;
	fv_tintii.push_back( 0.5f );
	fv_tintii.push_back( 0.5f );
	fv_tintii.push_back( 0.5f );
	input_instancing_untextured->s_uni_fv["tint"] = fv_tintii;
	input_shaders.push_back( input_instancing_untextured );

	sName = string( "tbo" );
	sVert = string( "shaders/drawTextureBufferVertex.glsl" );
	sFrag = string( "shaders/drawTextureBufferFragment.glsl" );
	InputShader* input_tbo = new InputShader( sName );
	input_tbo->s_vert = sVert;
	input_tbo->s_frag = sFrag;
	input_tbo->s_uni_i["posTextureBuffer"]	= 0;
	input_tbo->s_uni_f["PLANE_SCALE"]		= PLANE_SCALE;
	input_shaders.push_back( input_tbo );

	sName = string( "instancing_culled" );
	sVert = string( "shaders/instancing_culled.vert" );
	sFrag = string( "shaders/instancing_culled.frag" );
	InputShader* input_instancing_culled = new InputShader( sName );
	input_instancing_culled->s_vert = sVert;
	input_instancing_culled->s_frag = sFrag;
	input_instancing_culled->s_uni_i["diffuseTexture"] = 0;
	input_instancing_culled->s_uni_i["posTextureBuffer"] = 1;
	input_shaders.push_back( input_instancing_culled );

#ifdef CUDA_PATHS
#ifdef DEMO_SHADER
	err_manager->getError ("BEGIN: instancing_culled_rigged");
	sName = string( "instancing_culled_rigged" );
	sVert = string( "shaders/instancing_culled_rigged_demo_cuda.vert" );
	sFrag = string( "shaders/instancing_culled_rigged_demo_cuda.frag" );
	InputShader* input_instancing_culled_rigged = new InputShader( sName );
	input_instancing_culled_rigged->s_vert = sVert;
	input_instancing_culled_rigged->s_frag = sFrag;

	input_instancing_culled_rigged->s_uni_i["globalMT"]					= 0;		// clothing multi texture.
	input_instancing_culled_rigged->s_uni_i["torsoMT"]					= 1;		// clothing multi texture.
	input_instancing_culled_rigged->s_uni_i["legsMT"]					= 2;		// clothing multi texture.
	input_instancing_culled_rigged->s_uni_i["riggingMT"]				= 3;		// rigging multi texture.
	input_instancing_culled_rigged->s_uni_i["animationMT"]				= 4;		// animation multi texture.
	input_instancing_culled_rigged->s_uni_i["facialMT"]					= 5;		// facial multi texture.
	input_instancing_culled_rigged->s_uni_i["idsTextureBuffer"]			= 6;
	input_instancing_culled_rigged->s_uni_i["posTextureBuffer"]			= 7;
	input_instancing_culled_rigged->s_uni_i["color_table"]				= 10;
	input_instancing_culled_rigged->s_uni_i["pattern_table"]			= 11;

	input_instancing_culled_rigged->s_uni_i["AGENTS_NPOT"]				= 0;		//AGENTS_NPOT;
	input_instancing_culled_rigged->s_uni_i["ANIMATION_LENGTH"]			= 0;
	input_instancing_culled_rigged->s_uni_i["STEP"]						= 0;
	input_instancing_culled_rigged->s_uni_f["lod"]						= 0.0f;
	input_instancing_culled_rigged->s_uni_f["modelScale"]				= agent_scale;

	input_instancing_culled_rigged->s_uni_f["doHeightAndDisplacement"]	= 0.0f;
	input_instancing_culled_rigged->s_uni_f["doColor"]					= 0.0f;
	input_instancing_culled_rigged->s_uni_f["doPatterns"]				= 0.0f;
	input_instancing_culled_rigged->s_uni_f["doFacial"]					= 0.0f;

	input_shaders.push_back( input_instancing_culled_rigged );
	err_manager->getError ("END: instancing_culled_rigged");
#else
	err_manager->getError ("BEGIN: instancing_culled_rigged");
	sName = string( "instancing_culled_rigged" );
	sVert = string( "shaders/instancing_culled_rigged_demo.vert" );
	sFrag = string( "shaders/instancing_culled_rigged_demo.frag" );
	InputShader* input_instancing_culled_rigged = new InputShader( sName );
	input_instancing_culled_rigged->s_vert = sVert;
	input_instancing_culled_rigged->s_frag = sFrag;

	input_instancing_culled_rigged->s_uni_i["globalMT"]				= 0;		// clothing multi texture.
	input_instancing_culled_rigged->s_uni_i["torsoMT"]				= 1;		// clothing multi texture.
	input_instancing_culled_rigged->s_uni_i["legsMT"]				= 2;		// clothing multi texture.
	input_instancing_culled_rigged->s_uni_i["riggingMT"]			= 3;		// rigging multi texture.
	input_instancing_culled_rigged->s_uni_i["animationMT"]			= 4;		// animation multi texture.
	input_instancing_culled_rigged->s_uni_i["facialMT"]				= 5;		// facial multi texture.
	input_instancing_culled_rigged->s_uni_i["posTextureBuffer"]		= 6;
	input_instancing_culled_rigged->s_uni_i["posTexture"]			= 7;
	input_instancing_culled_rigged->s_uni_i["color_table"]			= 10;
	input_instancing_culled_rigged->s_uni_i["pattern_table"]		= 11;

	input_instancing_culled_rigged->s_uni_i["AGENTS_NPOT"]			= 0;		//AGENTS_NPOT;
	input_instancing_culled_rigged->s_uni_i["ANIMATION_LENGTH"]		= 0;
	input_instancing_culled_rigged->s_uni_i["STEP"]					= 0;
	input_instancing_culled_rigged->s_uni_f["lod"]					= 0.0f;
	input_instancing_culled_rigged->s_uni_f["modelScale"]			= agent_scale;

	input_shaders.push_back( input_instancing_culled_rigged );
	err_manager->getError ("END: instancing_culled_rigged");
#endif
#else
#ifdef DEMO_SHADER
	err_manager->getError ("BEGIN: instancing_culled_rigged");
	sName = string( "instancing_culled_rigged" );
	sVert = string( "shaders/instancing_culled_rigged_demo.vert" );
	sFrag = string( "shaders/instancing_culled_rigged_demo.frag" );
	InputShader* input_instancing_culled_rigged = new InputShader( sName );
	input_instancing_culled_rigged->s_vert = sVert;
	input_instancing_culled_rigged->s_frag = sFrag;

	input_instancing_culled_rigged->s_uni_i["globalMT"]					= 0;		// clothing multi texture.
	input_instancing_culled_rigged->s_uni_i["torsoMT"]					= 1;		// clothing multi texture.
	input_instancing_culled_rigged->s_uni_i["legsMT"]					= 2;		// clothing multi texture.
	input_instancing_culled_rigged->s_uni_i["riggingMT"]				= 3;		// rigging multi texture.
	input_instancing_culled_rigged->s_uni_i["animationMT"]				= 4;		// animation multi texture.
	input_instancing_culled_rigged->s_uni_i["facialMT"]					= 5;		// facial multi texture.
	input_instancing_culled_rigged->s_uni_i["posTextureBuffer"]			= 6;
	input_instancing_culled_rigged->s_uni_i["posTexture"]				= 7;
	input_instancing_culled_rigged->s_uni_i["color_table"]				= 10;
	input_instancing_culled_rigged->s_uni_i["pattern_table"]			= 11;

	input_instancing_culled_rigged->s_uni_i["AGENTS_NPOT"]				= 0;		//AGENTS_NPOT;
	input_instancing_culled_rigged->s_uni_i["ANIMATION_LENGTH"]			= 0;
	input_instancing_culled_rigged->s_uni_i["STEP"]						= 0;
	input_instancing_culled_rigged->s_uni_f["lod"]						= 0.0f;
	input_instancing_culled_rigged->s_uni_f["modelScale"]				= agent_scale;

	input_instancing_culled_rigged->s_uni_f["doHeightAndDisplacement"]	= 0.0f;
	input_instancing_culled_rigged->s_uni_f["doColor"]					= 0.0f;
	input_instancing_culled_rigged->s_uni_f["doPatterns"]				= 0.0f;
	input_instancing_culled_rigged->s_uni_f["doFacial"]					= 0.0f;

	input_shaders.push_back( input_instancing_culled_rigged );
	err_manager->getError ("END: instancing_culled_rigged");
#else
	err_manager->getError ("BEGIN: instancing_culled_rigged");
	sName = string( "instancing_culled_rigged" );
	sVert = string( "shaders/instancing_culled_rigged.vert" );
	sFrag = string( "shaders/instancing_culled_rigged.frag" );
	InputShader* input_instancing_culled_rigged = new InputShader( sName );
	input_instancing_culled_rigged->s_vert = sVert;
	input_instancing_culled_rigged->s_frag = sFrag;

	input_instancing_culled_rigged->s_uni_i["globalMT"]				= 0;		// clothing multi texture.
	input_instancing_culled_rigged->s_uni_i["torsoMT"]				= 1;		// clothing multi texture.
	input_instancing_culled_rigged->s_uni_i["legsMT"]				= 2;		// clothing multi texture.
	input_instancing_culled_rigged->s_uni_i["riggingMT"]			= 3;		// rigging multi texture.
	input_instancing_culled_rigged->s_uni_i["animationMT"]			= 4;		// animation multi texture.
	input_instancing_culled_rigged->s_uni_i["facialMT"]				= 5;		// facial multi texture.
	input_instancing_culled_rigged->s_uni_i["posTextureBuffer"]		= 6;
	input_instancing_culled_rigged->s_uni_i["posTexture"]			= 7;
	input_instancing_culled_rigged->s_uni_i["color_table"]			= 10;
	input_instancing_culled_rigged->s_uni_i["pattern_table"]		= 11;

	input_instancing_culled_rigged->s_uni_i["AGENTS_NPOT"]			= 0;		//AGENTS_NPOT;
	input_instancing_culled_rigged->s_uni_i["ANIMATION_LENGTH"]		= 0;
	input_instancing_culled_rigged->s_uni_i["STEP"]					= 0;
	input_instancing_culled_rigged->s_uni_f["lod"]					= 0.0f;
	input_instancing_culled_rigged->s_uni_f["modelScale"]			= agent_scale;

	input_shaders.push_back( input_instancing_culled_rigged );
	err_manager->getError ("END: instancing_culled_rigged");
#endif
#endif

#ifdef CUDA_PATHS
	err_manager->getError ("BEGIN: instancing_culled_rigged_shadow");
	sName = string( "instancing_culled_rigged_shadow" );
	sVert = string( "shaders/instancing_culled_rigged_shadow_cuda.vert" );
	sFrag = string( "shaders/instancing_culled_rigged_shadow_cuda.frag" );
	InputShader* input_instancing_culled_rigged_shadow = new InputShader( sName );
	input_instancing_culled_rigged_shadow->s_vert = sVert;
	input_instancing_culled_rigged_shadow->s_frag = sFrag;

	input_instancing_culled_rigged_shadow->s_uni_i["riggingMT"]			= 3;		// rigging multi texture.
	input_instancing_culled_rigged_shadow->s_uni_i["animationMT"]		= 4;		// animation multi texture.
	input_instancing_culled_rigged_shadow->s_uni_i["idsTextureBuffer"]	= 6;
	input_instancing_culled_rigged_shadow->s_uni_i["posTextureBuffer"]	= 7;

	input_instancing_culled_rigged_shadow->s_uni_i["AGENTS_NPOT"]		= 0;		//AGENTS_NPOT;
	input_instancing_culled_rigged_shadow->s_uni_i["ANIMATION_LENGTH"]	= 0;
	input_instancing_culled_rigged_shadow->s_uni_i["STEP"]				= 0;
	input_instancing_culled_rigged_shadow->s_uni_f["lod"]				= 0.0f;
	input_instancing_culled_rigged_shadow->s_uni_f["modelScale"]		= agent_scale;

	input_shaders.push_back( input_instancing_culled_rigged_shadow );
	err_manager->getError ("END: instancing_culled_rigged_shadow");
#else
	err_manager->getError ("BEGIN: instancing_culled_rigged_shadow");
	sName = string( "instancing_culled_rigged_shadow" );
	sVert = string( "shaders/instancing_culled_rigged_shadow.vert" );
	sFrag = string( "shaders/instancing_culled_rigged_shadow.frag" );
	InputShader* input_instancing_culled_rigged_shadow = new InputShader( sName );
	input_instancing_culled_rigged_shadow->s_vert = sVert;
	input_instancing_culled_rigged_shadow->s_frag = sFrag;

	input_instancing_culled_rigged_shadow->s_uni_i["riggingMT"]			= 3;		// rigging multi texture.
	input_instancing_culled_rigged_shadow->s_uni_i["animationMT"]		= 4;		// animation multi texture.
	input_instancing_culled_rigged_shadow->s_uni_i["posTextureBuffer"]	= 6;
	input_instancing_culled_rigged_shadow->s_uni_i["posTexture"]		= 7;

	input_instancing_culled_rigged_shadow->s_uni_i["AGENTS_NPOT"]		= 0;		//AGENTS_NPOT;
	input_instancing_culled_rigged_shadow->s_uni_i["ANIMATION_LENGTH"]	= 0;
	input_instancing_culled_rigged_shadow->s_uni_i["STEP"]				= 0;
	input_instancing_culled_rigged_shadow->s_uni_f["lod"]				= 0.0f;
	input_instancing_culled_rigged_shadow->s_uni_f["modelScale"]		= agent_scale;

	input_shaders.push_back( input_instancing_culled_rigged_shadow );
	err_manager->getError ("END: instancing_culled_rigged_shadow");
#endif

	sName = string( "instancing_culled_rigged_low" );
	sVert = string( "shaders/instancing_culled_rigged_low.vert" );
	sFrag = string( "shaders/instancing_culled_rigged_low.frag" );
	InputShader* input_instancing_culled_rigged_low = new InputShader( sName );
	input_instancing_culled_rigged_low->s_vert = sVert;
	input_instancing_culled_rigged_low->s_frag = sFrag;
	input_instancing_culled_rigged_low->s_uni_i["diffuseTexture"]	= 0;
	input_instancing_culled_rigged_low->s_uni_i["zonesTexture"]		= 1;
	input_instancing_culled_rigged_low->s_uni_i["weightsTexture"]	= 2;
	input_instancing_culled_rigged_low->s_uni_i["posTextureBuffer"] = 3;
	input_shaders.push_back( input_instancing_culled_rigged_low );

#ifdef CUDA_PATHS
	sName = string( "vfc_lod_assignment" );
	sVert = string( "shaders/VFCLodAssigmentTexVertexCuda.glsl" );
	sGeom = string( "shaders/VFCLodAssigmentTexGeometryCuda.glsl" );
	InputShader* input_vfc_lod_ass = new InputShader( sName );
	input_vfc_lod_ass->is_transform_feedback = true;
	input_vfc_lod_ass->s_vert = sVert;
	input_vfc_lod_ass->s_geom = sGeom;
	input_vfc_lod_ass->transform_feedback_vars.push_back( str_gl_Position );
	input_vfc_lod_ass->s_uni_i["idTex"]		= 2;
	input_vfc_lod_ass->s_uni_i["positions"] = 7;
	input_vfc_lod_ass->s_uni_f["modelScale"]= agent_scale;
	input_vfc_lod_ass->s_ipri = GL_POINTS;
	input_vfc_lod_ass->s_opri = GL_POINTS;
	input_shaders.push_back( input_vfc_lod_ass );
#else
	sName = string( "vfc_lod_assignment" );
	sVert = string( "shaders/VFCLodAssigmentTexVertex.glsl" );
	sGeom = string( "shaders/VFCLodAssigmentTexGeometry.glsl" );
	InputShader* input_vfc_lod_ass = new InputShader( sName );
	input_vfc_lod_ass->is_transform_feedback = true;
	input_vfc_lod_ass->s_vert = sVert;
	input_vfc_lod_ass->s_geom = sGeom;
	input_vfc_lod_ass->transform_feedback_vars.push_back( str_gl_Position );
	input_vfc_lod_ass->s_uni_i["positions"] = 0;
	input_vfc_lod_ass->s_uni_i["idTex"]		= 2;
	input_vfc_lod_ass->s_uni_f["modelScale"]= agent_scale;
	input_vfc_lod_ass->s_ipri = GL_POINTS;
	input_vfc_lod_ass->s_opri = GL_POINTS;
	input_shaders.push_back( input_vfc_lod_ass );
#endif

	sName = string( "vfc_lod_selection" );
	sVert = string( "shaders/VFCLodAssigmentTexVertex.glsl" );
	sGeom = string( "shaders/lodSelectionGeometry.glsl" );
	InputShader* input_vfc_lod_sel = new InputShader( sName );
	input_vfc_lod_sel->is_transform_feedback = true;
	input_vfc_lod_sel->s_vert = sVert;
	input_vfc_lod_sel->s_geom = sGeom;
	input_vfc_lod_sel->transform_feedback_vars.push_back( str_gl_Position );
	input_vfc_lod_sel->s_ipri = GL_POINTS;
	input_vfc_lod_sel->s_opri = GL_POINTS;
	input_shaders.push_back( input_vfc_lod_sel );

	sName = string( "pass_rect" );
	sVert = string( "shaders/passthru_rect_vertex.glsl" );
	sFrag = string( "shaders/passthru_rect_fragment.glsl" );
	InputShader* input_pass_rect = new InputShader( sName );
	input_pass_rect->s_vert = sVert;
	input_pass_rect->s_frag = sFrag;
	input_pass_rect->s_uni_i["tex"] = 0;
	input_shaders.push_back( input_pass_rect );

	sName = string( "pass_2d" );
	sVert = string( "shaders/passthru_2d_vertex.glsl" );
	sFrag = string( "shaders/passthru_2d_fragment.glsl" );
	InputShader* input_pass_2d = new InputShader( sName );
	input_pass_2d->s_vert = sVert;
	input_pass_2d->s_frag = sFrag;
	input_pass_2d->s_uni_i["tex"] = 0;
	input_shaders.push_back( input_pass_2d );

	sName = string( "clothing_coordinate" );
	sVert = string( "shaders/passthru_2d_vertex.glsl" );
	sFrag = string( "shaders/cloth_coordinate.frag" );
	InputShader* input_cloth_coordinate = new InputShader( sName );
	input_cloth_coordinate->s_vert = sVert;
	input_cloth_coordinate->s_frag = sFrag;
	input_cloth_coordinate->s_uni_i["pattern_tex"]		= 0;
	input_cloth_coordinate->s_uni_i["coordinate_tex"]	= 1;
	input_cloth_coordinate->s_uni_i["wrinkle_tex"]		= 2;
	input_shaders.push_back( input_cloth_coordinate );

	err_manager->getError( "Before init shaders:" );
	if( !glsl_manager->init( input_shaders ) )
	{
		log_manager->log( LogManager::LERROR, "While initializing GLSL shaders!" );
		cleanup();
		exit( 1 );
	}
	log_manager->log( LogManager::INFORMATION, "GLSL SHADERS initialization complete." );
}
//
//=======================================================================================
//
void postconfigure_shaders( void )
{

	glsl_manager->activate( "mdp_floor" );
	{
		float tH = 1000.0f / (float)obstacle_manager->getSceneDepthInTiles();
		float tR = tH / sqrtf( 3.0f );
		float tS = 3.0f*tR/2.0f;
		glsl_manager->setUniformf( "mdp_floor", (char*)str_tile_side.c_str(),     tS);
		glsl_manager->setUniformf( "mdp_floor", (char*)str_tile_height.c_str(),   tH);
		glsl_manager->setUniformf( "mdp_floor", (char*)str_policy_width.c_str(),  (float)obstacle_manager->getSceneWidthInTiles() );
		glsl_manager->setUniformf( "mdp_floor", (char*)str_policy_height.c_str(), (float)obstacle_manager->getSceneDepthInTiles() );
		if( policy_floor )
		{
			glsl_manager->setUniformf( "mdp_floor", (char*)str_policy_on.c_str(), 1.0f );
		}
		else
		{
			glsl_manager->setUniformf( "mdp_floor", (char*)str_policy_on.c_str(), 0.0f );
		}
		if( density_floor )
		{
			glsl_manager->setUniformf( "mdp_floor", (char*)str_density_on.c_str(), 1.0f );
		}
		else
		{
			glsl_manager->setUniformf( "mdp_floor", (char*)str_density_on.c_str(), 0.0f );
		}
	}
	glsl_manager->deactivate( "mdp_floor" );

#if defined MDPS_SQUARE_NOLCA || defined MDPS_SQUARE_LCA
	unsigned int swit = obstacle_manager->getSceneWidthInTiles();
	unsigned int sdit = obstacle_manager->getSceneDepthInTiles();
#if defined MDPS_SQUARE_LCA
	glsl_manager->activate( "mdp_floor" );
	{
		glsl_manager->setUniformf( "mdp_floor", "tile_width",	2000.0f / (float)swit ); // change the size of the arrows
		glsl_manager->setUniformf( "mdp_floor", "tile_height",	2000.0f / (float)sdit );
	}
	glsl_manager->deactivate( "mdp_floor" );
#elif defined MDPS_SQUARE_NOLCA
	glsl_manager->activate( "mdp_floor" );
	{
		glsl_manager->setUniformf( "mdp_floor", "tile_width",	1000.0f / (float)swit );
		glsl_manager->setUniformf( "mdp_floor", "tile_height",	1000.0f / (float)sdit );
	}
	glsl_manager->deactivate( "mdp_floor" );
#endif
	glsl_manager->activate( "mdp_floor" );
	{
		glsl_manager->setUniformf( "mdp_floor", "policy_width",		(float)swit );
		glsl_manager->setUniformf( "mdp_floor", "policy_height",	(float)sdit );
	}
	glsl_manager->deactivate( "mdp_floor" );
#endif
	log_manager->log( LogManager::INFORMATION, "GLSL SHADERS postconfiguration complete." );
}
//
//=======================================================================================
//
void init_vbos( void )
{
    vbo_manager = new VboManager							( 	log_manager,
    															err_manager,
    															glsl_manager 						);

	vbo_manager->setInstancingLocations						(	str_instancing_textured,
																str_positions_textured,
																str_rotations_textured,
																str_scales_textured,
																str_normal_textured,
																str_texCoord0_textured,
																str_ViewMat4x4_textured,
																str_instancing_untextured,
																str_positions_untextured,
																str_rotations_untextured,
																str_scales_untextured,
																str_normal_untextured,
																str_ViewMat4x4_untextured			);
	vbo_manager->setInstancingCulledLocations				(	str_instancing_culled,
																str_normal_textured,
																str_texCoord0_textured,
																str_ViewMat4x4_textured				);
	vbo_manager->setInstancingCulledRiggedLocations			(	str_instancing_culled_rigged,
																str_normalV,
																str_dt,
																str_texCoord0_textured,
																str_ViewMat4x4_textured				);
	vbo_manager->setInstancingCulledRiggedShadowLocations	(	str_instancing_culled_rigged_shadow,
																str_normalV,
																str_texCoord0_textured,
																str_dt,
																str_ViewMat4x4_textured,
																str_ProjMat4x4,
																str_ShadowMat4x4 					);
	if( err_manager->getError( "After setting instancing locations:" ) != GL_NO_ERROR )
	{
		log_manager->log( LogManager::LERROR, "While initializing VBOs!" );
		cleanup();
		exit( 1 );
	}
	log_manager->log( LogManager::INFORMATION, "VBOs initialization complete." );
}
//
//=======================================================================================
//
void init_crowds( void )
{
	crowd_manager	= new CrowdManager( log_manager,
										err_manager,
										fbo_manager,
										vbo_manager,
										glsl_manager,
										str_crowd_xml	);

	if( crowd_manager->init() == false )
	{
		log_manager->file_log( LogManager::LERROR, "While initializing CROWDS" );
		cleanup();
		exit( 1 );
	}

	if( crowd_manager->loadAssets() == false )
	{
		log_manager->file_log( LogManager::LERROR, "While loading crowd assets!" );
		cleanup();
		exit( 1 );
	}
	log_manager->log( LogManager::INFORMATION, "CROWDS initialization complete." );
}
//
//=======================================================================================
//
void init_fbos( void )
{
	vector<InputFbo*> fbo_inputs;

	vector<InputFboTexture*> input_fbo_mdp;
	InputFboTexture* mdp_tex1 = new InputFboTexture( str_mdp_tex, InputFbo::GPGPU );
	input_fbo_mdp.push_back( mdp_tex1 );
	InputFbo* fbo_mdp = new InputFbo( str_mdp_fbo, GL_TEXTURE_RECTANGLE, input_fbo_mdp, 2000, 2000 );
	fbo_inputs.push_back( fbo_mdp );

	vector<InputFboTexture*> input_fbo_textures1;
	InputFboTexture* tex1 = new InputFboTexture( str_display_tex,       InputFbo::TYPICAL			);
	InputFboTexture* tex2 = new InputFboTexture( str_display_depth_tex, InputFbo::DEPTH_NO_COMPARE	);
	input_fbo_textures1.push_back( tex1 );
	input_fbo_textures1.push_back( tex2 );
	InputFbo* fbo1 = new InputFbo( str_display_fbo, GL_TEXTURE_RECTANGLE, input_fbo_textures1, INIT_WINDOW_WIDTH, INIT_WINDOW_HEIGHT );
	fbo_inputs.push_back( fbo1 );

	vector<Crowd*> crowds = crowd_manager->getCrowds();
	for( unsigned int c = 0; c < crowds.size(); c++ )
	{
		vector<InputFbo*> crowdInputs;
		Crowd* crowd = crowds[c];
		crowd->initFboInputs( crowdInputs );
		for( unsigned int ci = 0; ci < crowdInputs.size(); ci++ )
		{
			fbo_inputs.push_back( crowdInputs[ci] );
		}
	}

	fbo_manager = new FboManager( log_manager, glsl_manager, fbo_inputs );
	if( !fbo_manager->init() )
	{
		log_manager->log( LogManager::LERROR, "While initializing FBOs!" );
		cleanup();
		exit( 1 );
	}
	crowd_manager->setFboManager( fbo_manager );

	GLuint  agents_offset	= 0;
	GLuint  group_id		= 0;
	for( unsigned int c = 0; c < crowds.size(); c++ )
	{
		Crowd* crowd = crowds[c];
		crowd->init_ids( group_id, agents_offset );
	}
	fbo_manager->report();
	log_manager->log( LogManager::INFORMATION, "FBOs initialization complete." );
}
//
//=======================================================================================
//
void init_textures( void )
{
    TextureManager::getInstance()->init( err_manager, log_manager );
    log_manager->log( LogManager::INFORMATION, "TEXTURES initialization complete." );
}
//
//=======================================================================================
//
void init_skybox( void )
{
	vector<bool> skb_bump;
	skb_bump.push_back( false );
	skb_bump.push_back( false );
	skb_bump.push_back( false );
	skb_bump.push_back( false );
	skb_bump.push_back( false );
	skb_bump.push_back( false );
	vector<float> skb_tile;
	skb_tile.push_back( 1.0f );		//FRONT
	skb_tile.push_back( 1.0f );		//BACK
	skb_tile.push_back( 1.0f );		//LEFT
	skb_tile.push_back( 1.0f );		//RIGHT
	skb_tile.push_back( 1.0f );		//TOP
	skb_tile.push_back( 1.0f );		//BOTTOM
	glm::vec3 vDisp( 0.0f, PLANE_SCALE / 6.0f, 0.0f );
	glm::vec3 vDims( 5.0f * PLANE_SCALE,
					 5.0f * PLANE_SCALE,
					 5.0f * PLANE_SCALE );
	skybox_manager = new SkyboxManager( 1,
										glsl_manager,
										vbo_manager,
										log_manager,
										err_manager,
										vDisp,
										vDims,
										skb_bump,
										skb_tile,
										false,
										false 		);
	bool skb_tex =
	skybox_manager->LoadSkyboxTextures( str_skybox_front,
										GL_REPLACE,
										str_skybox_back,
										GL_REPLACE,
										str_skybox_left,
										GL_REPLACE,
										str_skybox_right,
										GL_REPLACE,
										str_skybox_top,
										GL_REPLACE,
										str_skybox_bottom,
										GL_REPLACE			);
	if( skb_tex == false )
	{
		log_manager->log( LogManager::LERROR, "While loading Skybox textures!" );
		cleanup();
		exit( 1 );
	}
	log_manager->log( LogManager::INFORMATION, "MODEL::Skybox ready." );
}
//
//=======================================================================================
//
void init_models( void )
{
//->AXES
    axes = new Axes( 100.0f );
    log_manager->log( LogManager::INFORMATION, "MODEL::Axes ready." );
//<-AXES

//->SKYBOX
    init_skybox();
//<-SKYBOX
    log_manager->log( LogManager::INFORMATION, "MODELS initialization complete." );
}
//
//=======================================================================================
//
void init_scenario( void )
{
#ifdef DRAW_SCENARIO

	if( scenario_type == ST_O1P2 )
	{
		str_scenario_path	= string( "assets/o1p2/"		);
		str_scenario_obj	= string( "o1p2.obj"			);
		//str_scenario_obj	= string( "o1p2_furnished.obj"	);
	}
	else if( scenario_type == ST_EIFFEL )
	{

		str_scenario_path	= string( "assets/eiffel/"		);
		str_scenario_obj	= string( "EiffelTower.obj"		);
		//str_scenario_path	= string( "assets/eiffel/Eiffel2Obj/"	);
		//str_scenario_obj	= string( "Eiffel2.obj"					);
		//str_scenario_obj	= string( "Eiffel2_earth.obj"			);
	}
	else if( scenario_type == ST_TOWN )
	{
		str_scenario_path	= string( "assets/town/"		);
		str_scenario_obj	= string( "town.obj"			);
	}
	else if( scenario_type == ST_MAZE )
	{
		//return;
		str_scenario_path	= string( "assets/base/"		);
		str_scenario_obj	= string( "base.obj"			);
	}
	else if( scenario_type == ST_CCM )
	{
		str_scenario_path	= string( "assets/ccm/"				);
		//str_scenario_obj	= string( "ccm_earth.obj"			);
		//str_scenario_obj	= string( "ccm_earth_notrees.obj"	);
		//str_scenario_obj	= string( "ccm_map.obj"				);
		str_scenario_obj	= string( "ccm_map_notrees.obj"		);
		//str_scenario_obj	= string( "ccm_mdp_notrees.obj"		);
	}
	scenario = new Scenario	(	str_scenario_obj,
								str_scenario_path,
								vbo_manager,
								glsl_manager,
								str_scenario_obj,
								1.0f				);
	log_manager->log( LogManager::INFORMATION, "SCENARIO MODEL initialization complete." );
#endif
}
//
//=======================================================================================
//
#include "cShadows.h"
void init_lights( void )
{
	float global_ambient[]	= {    0.12f,	  0.12f,	0.12f,	1.0f };

	//float position0[]		= {    PLANE_SCALE*2.0f, PLANE_SCALE*2.0f, -PLANE_SCALE*2.0f,	1.0f };
	float position0[]		= {    PLANE_SCALE, PLANE_SCALE*2.0f, -PLANE_SCALE,	1.0f };
	shadowMatrix( position0 );
	log_manager->log( LogManager::CONFIGURATION, 	"Shadow Matrix at position (%.3f, %.3f, %.3f) ready.",
													position0[0],
													position0[1],
													position0[2] 										);
	float diffuse0[]		= {    1.0f,	  1.0f,		1.0f,	1.0f };
	float specular0[]		= {    0.4f,	  0.4f,		0.4f,	1.0f };
	float ambient0[]		= {    0.2f,	  0.2f,		0.2f,	1.0f };

	//float position1[]		= {	   PLANE_SCALE*2.0f, PLANE_SCALE*2.0f, PLANE_SCALE*2.0f, 1.0f };
	float position1[]		= {	   PLANE_SCALE, PLANE_SCALE, PLANE_SCALE, 1.0f };
	float diffuse1[]		= {    1.0f,	  1.0f,		1.0f,	1.0f };
	float specular1[]		= {    0.4f,	  0.4f,		0.4f,	1.0f };
	float ambient1[]		= {    0.2f,	  0.2f,		0.2f,	1.0f };

	float position2[]		= {	   -PLANE_SCALE*2.0f, PLANE_SCALE*2.0f, -PLANE_SCALE*2.0f, 1.0f };
	float diffuse2[]		= {    1.0f,	  1.0f,		1.0f,	1.0f };
	float specular2[]		= {    0.4f,	  0.4f,		0.4f,	1.0f };
	float ambient2[]		= {    0.2f,	  0.2f,		0.2f,	1.0f };

	float position3[]		= {    -PLANE_SCALE*2.0f, PLANE_SCALE*2.0f, PLANE_SCALE*2.0f, 1.0f };
	float diffuse3[]		= {    1.0f,	  1.0f,		1.0f,	1.0f };
	float specular3[]		= {    0.4f,	  0.4f,		0.4f,	1.0f };
	float ambient3[]		= {    0.2f,	  0.2f,		0.2f,	1.0f };
	//glLightModelfv( GL_LIGHT_MODEL_AMBIENT, global_ambient );
	//glLightModeli( GL_LIGHT_MODEL_COLOR_CONTROL, GL_SEPARATE_SPECULAR_COLOR );
/*
	Source: http://www.csit.parkland.edu/~dbock/Portfolio/Content/Lights.html
	GL_LIGHT_MODEL_LOCAL_VIEWER:
	GL_FALSE: infinite viewpoint - viewing vector doesn't change across surface - faster,
			  less realistic
    GL_TRUE:  local viewpoint - viewing vector changes across surface - slower,
			  more realistic
*/
	//glLightModeli( GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE );

	glLightfv(  GL_LIGHT0,  GL_POSITION,    position0	);
	glLightfv(  GL_LIGHT0,  GL_DIFFUSE,     diffuse0	);
	glLightfv(  GL_LIGHT0,  GL_SPECULAR,    specular0	);
	glLightfv(  GL_LIGHT0,  GL_AMBIENT,     ambient0	);
	glEnable (  GL_LIGHT0		                        );

	glLightfv(  GL_LIGHT1,  GL_POSITION,    position1	);
	glLightfv(  GL_LIGHT1,  GL_DIFFUSE,     diffuse1	);
	glLightfv(  GL_LIGHT1,  GL_SPECULAR,    specular1	);
	glLightfv(  GL_LIGHT1,  GL_AMBIENT,     ambient1	);
	glEnable (  GL_LIGHT1		                        );

	glLightfv(  GL_LIGHT2,  GL_POSITION,    position2	);
	glLightfv(  GL_LIGHT2,  GL_DIFFUSE,     diffuse2	);
	glLightfv(  GL_LIGHT2,  GL_SPECULAR,    specular2	);
	glLightfv(  GL_LIGHT2,  GL_AMBIENT,     ambient2	);
	glEnable (  GL_LIGHT2		                        );

	glLightfv(  GL_LIGHT3,  GL_POSITION,    position3	);
	glLightfv(  GL_LIGHT3,  GL_DIFFUSE,     diffuse3	);
	glLightfv(  GL_LIGHT3,  GL_SPECULAR,    specular3	);
	glLightfv(  GL_LIGHT3,  GL_AMBIENT,     ambient3	);
	glEnable (  GL_LIGHT3		                        );

	glEnable (  GL_LIGHTING	                            );
	log_manager->log( LogManager::INFORMATION, "LIGHTS initialization complete." );
}
//
//=======================================================================================
//
void init_cameras( void )
{
	glm::vec3 p0;   glm::vec3 p1;   glm::vec3 p2;   glm::vec3 p3;   glm::vec3 p4;
	glm::vec3 d0;   glm::vec3 d1;   glm::vec3 d2;   glm::vec3 d3;   glm::vec3 d4;
	glm::vec3 r0;   glm::vec3 r1;   glm::vec3 r2;   glm::vec3 r3;   glm::vec3 r4;
	glm::vec3 u0;   glm::vec3 u1;   glm::vec3 u2;   glm::vec3 u3;   glm::vec3 u4;

	p0.x = -PLANE_SCALE;    p0.y =  PLANE_SCALE/2.0f;   p0.z =  PLANE_SCALE;
	r0.x =  PLANE_SCALE;    r0.y =  0.0f;               r0.z =  PLANE_SCALE;
	p1.x =  PLANE_SCALE;    p1.y =  PLANE_SCALE/2.0f;   p1.z =  PLANE_SCALE;
	r1.x =  PLANE_SCALE;    r1.y =  0.0f;               r1.z = -PLANE_SCALE;
	p2.x =  PLANE_SCALE;    p2.y =  PLANE_SCALE/2.0f;   p2.z = -PLANE_SCALE;
	r2.x = -PLANE_SCALE;    r2.y =  0.0f;               r2.z = -PLANE_SCALE;
	p3.x = -PLANE_SCALE;    p3.y =  PLANE_SCALE/2.0f;   p3.z = -PLANE_SCALE;
	r3.x = -PLANE_SCALE;    r3.y =  0.0f;               r3.z =  PLANE_SCALE;
	p4.x =  0.0f;           p4.y =  PLANE_SCALE*2.0f;   p4.z =  0.0f;
	r4.x =  1.0f;           r4.y =  0.0f;               r4.z =  0.0f;

	d0 = -p0;   d1 = -p1;   d2 = -p2;   d3 = -p3;   d4 = -p4;

	d0 = glm::normalize( d0 );  d1 = glm::normalize( d1 );
	d2 = glm::normalize( d2 );  d3 = glm::normalize( d3 );
	d4 = glm::normalize( d4 );
	r0 = glm::normalize( r0 );  r1 = glm::normalize( r1 );
	r2 = glm::normalize( r2 );  r3 = glm::normalize( r3 );
	r4 = glm::normalize( r4 );
	u0 = glm::cross( r0, d0 );  u1 = glm::cross( r1, d1 );
	u2 = glm::cross( r2, d2 );  u3 = glm::cross( r3, d3 );
	u4 = glm::cross( r4, d4 );

	predef_cam_pos.push_back( p0 ); predef_cam_pos.push_back( p1 );
	predef_cam_pos.push_back( p2 ); predef_cam_pos.push_back( p3 );
	predef_cam_pos.push_back( p4 );
	predef_cam_dir.push_back( d0 ); predef_cam_dir.push_back( d1 );
	predef_cam_dir.push_back( d2 ); predef_cam_dir.push_back( d3 );
	predef_cam_dir.push_back( d4 );
	predef_cam_up.push_back( u0 );  predef_cam_up.push_back( u1 );
	predef_cam_up.push_back( u2 );  predef_cam_up.push_back( u3 );
	predef_cam_up.push_back( u4 );

    camera1 = new Camera( 0, Camera::FREE, Frustum::RADAR );
    camera2 = new Camera( 1, Camera::FREE, Frustum::RADAR );
    camera  = camera2;

    vec3 pos1( PLANE_SCALE / 20.0f,  PLANE_SCALE / 20.0f,  -PLANE_SCALE / 20.0f );
    vec3 dir1( 0.0f, 0.0f, -1.0f );
    vec3 up1( 0.0f, 1.0f, 0.0f );
    vec3 piv1 = pos1 + dir1;

    vec3 pos2( -PLANE_SCALE / 2.0f,  PLANE_SCALE / 2.0f,  PLANE_SCALE / 2.0f );
    vec3 dir2( -pos2.x, -pos2.y, -pos2.z );
    vec3 up2( 0.0f, 1.0f, 0.0f );
    vec3 piv2 = pos2 + dir2;

    camera1->getFrustum()->setFovY( 54.43f );
    camera1->getFrustum()->setNearD( 1.0f );
    camera1->getFrustum()->setFarD( 5.0f * PLANE_SCALE );
    camera1->setEyeSeparation( 35.0f );
    camera1->setPivot( piv1 );
    camera1->setPosition( pos1 );
    camera1->setUpVec( up1 );
    camera1->setDirection( dir1 );

    camera2->getFrustum()->setFovY( 54.43f );
    camera2->getFrustum()->setNearD( 1.0f );
    camera2->getFrustum()->setFarD( 5.0f * PLANE_SCALE );
    camera2->setEyeSeparation( 35.0f );
    camera2->setPivot( piv2 );
    camera2->setPosition( pos2 );
    camera2->setUpVec( up2 );
    camera2->setDirection( dir2 );

    camNear = camera2->getFrustum()->getNearD();
    log_manager->log( LogManager::INFORMATION, "CAMERAS initialization complete." );
}
//
//=======================================================================================
//
void init_obstacles( void )
{
	obstacle_manager = new ObstacleManager( glsl_manager,
											vbo_manager,
											log_manager,
											PLANE_SCALE * 2.0f,
											PLANE_SCALE * 2.0f	);
	obstacle_manager->init(	obstacle_type,
							str_mdp_csv,
							fbo_manager		);

	vector<glm::vec2> occupation;
	vector<vector<GROUP_FORMATION>> formations;
	vector<GROUP_FORMATION> formation;

	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_CIRCLE );
	/*
	formation.push_back( GFRM_SQUARE );
	formation.push_back( GFRM_SQUARE );
	formation.push_back( GFRM_SQUARE );
	formation.push_back( GFRM_SQUARE );
	formation.push_back( GFRM_SQUARE );
	formation.push_back( GFRM_SQUARE );
	formation.push_back( GFRM_SQUARE );
	formation.push_back( GFRM_SQUARE );
	formation.push_back( GFRM_SQUARE );
	formation.push_back( GFRM_SQUARE );
	*/
	/*
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_SQUARE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_SQUARE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_SQUARE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_SQUARE );
	formation.push_back( GFRM_CIRCLE );
	formation.push_back( GFRM_SQUARE );
	*/
	formations.push_back( formation );
	crowd_manager->initPaths(	crowd_position,
								scenario_type,
								str_scenario_speed,
								PLANE_SCALE,
								obstacle_manager->getPolicies(),
								obstacle_manager->getSceneWidthInTiles(),
								obstacle_manager->getSceneDepthInTiles(),
								occupation,
								formations								);
	log_manager->log( LogManager::INFORMATION, "OBSTACLES initialization complete." );
}
//
//=======================================================================================
//
void init_gl( int argc, char *argv[] )
{
    glutInit					( 	&argc,
    								argv						);
    glutInitDisplayMode			( 	GLUT_RGBA 	|
    								GLUT_DEPTH 	|
    								GLUT_DOUBLE 				);
    glutInitWindowSize			( 	INIT_WINDOW_WIDTH,
    								INIT_WINDOW_HEIGHT			);
    glutInitWindowPosition		( 	10,
    								10 							);
	glutInitContextVersion		(	3,
									1							);
	glutCreateWindow        	(   (char*)str_title.c_str() 	);
	glutInitContextFlags		(	GLUT_FORWARD_COMPATIBLE		);
	glutInitContextProfile		(	GLUT_COMPATIBILITY_PROFILE	);

	glewExperimental 	= GL_TRUE;
	int glew_status 	= glewInit();
    if( glew_status != GLEW_OK )
    {
		log_manager->log( LogManager::LERROR, "GLEW initialization failed!" );
		cleanup();
		exit( 1 );
    }

	glClearColor    (   195.0f/255.0f,
                        195.0f/255.0f,
                        195.0f/255.0f,
                        0.0                             );
	glShadeModel    (   GL_SMOOTH                       );
	glEnable        (   GL_DEPTH_TEST                   );
	glEnable        (   GL_TEXTURE_2D                   );
	glHint          (   GL_PERSPECTIVE_CORRECTION_HINT,
                        GL_NICEST                       );
	glEnable        (   GL_MULTISAMPLE                  );
	glEnable        (   GL_BLEND                        );
	glBlendFunc     (   GL_SRC_ALPHA,
                        GL_ONE_MINUS_SRC_ALPHA          );

    string gl_vdr( (const char*)glGetString     ( GL_VENDOR                     ) );
    string gl_ren( (const char*)glGetString     ( GL_RENDERER                   ) );
    string gl_ver( (const char*)glGetString     ( GL_VERSION                    ) );
    string gw_ver( (const char*)glewGetString   ( GLEW_VERSION                  ) );
    string sl_ver( (const char*)glGetString     ( GL_SHADING_LANGUAGE_VERSION   ) );

	log_manager->log( LogManager::CONTEXT, "Vendor:   %s",	    gl_vdr.c_str() );
	log_manager->log( LogManager::CONTEXT, "Renderer: %s",	    gl_ren.c_str() );
	log_manager->log( LogManager::CONTEXT, "GL:       %s",	    gl_ver.c_str() );
	log_manager->log( LogManager::CONTEXT, "GLEW:     %s",      gw_ver.c_str() );
	log_manager->log( LogManager::CONTEXT, "GLSL:     %s",      sl_ver.c_str() );
	log_manager->separator();
}
//
//=======================================================================================
//
void init( int argc, char *argv[] )
{
	char cCurrentPath[1000];
#ifdef __unix
	getcwd( cCurrentPath, sizeof(cCurrentPath) );
#elif defined _WIN32
	_getcwd( cCurrentPath, sizeof(cCurrentPath) );
#endif
	str_curr_dir = string( cCurrentPath );

	Timer::getInstance()->init();

	init_gl( argc, argv );

	err_manager	= new GlErrorManager( log_manager );

	float init_cpu;
#if defined _WIN32
	DWORD start;
	Timer::getInstance()->start( start );
#elif defined __unix
	timeval start;
	Timer::getInstance()->start( start );
#endif
	log_manager->log( LogManager::INFORMATION, "Initializing GLOBAL VARIABLES..." );
    init_globals();
    log_manager->separator();

	log_manager->log( LogManager::INFORMATION, "Initializing LIGHTS..." );
    init_lights();
    log_manager->separator();

    log_manager->log( LogManager::INFORMATION, "Initializing TEXTURES..." );
	init_textures();
	log_manager->separator();

	log_manager->log( LogManager::INFORMATION, "Initializing GLSL SHADERS..." );
    init_glsl();
    log_manager->separator();

    log_manager->log( LogManager::INFORMATION, "Initializing VBOs..." );
    init_vbos();
    log_manager->separator();

    log_manager->log( LogManager::INFORMATION, "Initializing CROWDS..." );
    init_crowds();
    log_manager->separator();

    log_manager->log( LogManager::INFORMATION, "Initializing FBOs..." );
    init_fbos();
    log_manager->separator();

    log_manager->log( LogManager::INFORMATION, "Initializing CAMERAS..." );
    init_cameras();
    log_manager->separator();

    log_manager->log( LogManager::INFORMATION, "Initializing MODELS..." );
    init_models();
	init_scenario();
    log_manager->separator();

    log_manager->log( LogManager::INFORMATION, "Initializing OBSTACLES..." );
    init_obstacles();
    log_manager->separator();

	log_manager->log( LogManager::INFORMATION, "Postconfiguring GLSL SHADERS..." );
	postconfigure_shaders();
	log_manager->separator();

#if defined _WIN32
	init_cpu = Timer::getInstance()->stop( start );
#elif defined __unix
	init_cpu = Timer::getInstance()->stop( start );
#endif
    log_manager->file_log( LogManager::INFORMATION, "<B>CASIM</B> Initialized in <B>%08.5f</B> seconds", init_cpu/1000.0f );
    log_manager->console_log( LogManager::INFORMATION, "CASIM Initialized in %08.5f seconds", init_cpu/1000.0f );
    log_manager->separator();
}
//
//=======================================================================================
