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
//->FLOAT
float               PLANE_SCALE             = 10000.0f;
float			    fps					    = 0.0f;
float			    delta_time			    = 0.0f;
float				time_counter			= 0.0f;
float			    prev_time			    = 0.0f;
float 				camNear					= 0.0f;
float				camAccel				= 100.0f;
float				one_sixtieth			= 1.0f / 60.0f;
float				one_thirtieth			= 1.0f / 30.0f;
float				agent_scale				= 1.0f;

float				total_mdp_process		= 0.0f;
float				total_mdp_iteration		= 0.0f;
float				delta_time_avg			= 0.0f;
float				delta_time_mdp_avg		= 0.0f;
float				init_mdp_structures		= 0.0f;
float				init_mdp_perms			= 0.0f;
float				uploading_mdp			= 0.0f;
float				iterating_mdp_avg		= 0.0f;
int					mdp_iterations			= 0;
float				downloading_mdp			= 0.0f;
float				update_mdp_policy		= 0.0f;

//<-FLOAT
//=======================================================================================
//->INT
int				    update_frame		    = 0;
int					glut_mod				= 0;
//<-INT
//=======================================================================================
//->BOOL
bool				lMouseDown				= false;
bool				rMouseDown				= false;
bool				mMouseDown				= false;
bool			    ctrlDown		        = false;

bool				runpaths				= false;
bool				showstats				= true;
bool				animating				= false;
bool				wireframe				= false;
bool				hideCharacters			= false;
bool				drawShadows				= false;
bool				policy_floor			= true;
bool				density_floor			= true;
bool				spring_scenario			= true;

#ifdef DEMO_SHADER
bool				doHeightAndDisplacement	= false;
bool				doPatterns				= false;
bool				doColor					= false;
bool				doFacial				= false;
#endif
//<-BOOL
//=======================================================================================
//->UNSIGNED
unsigned long	    frame_counter		    = 0;

unsigned int		predef_cam_index		= 0;
unsigned int 		d_lod1 					= 0;
unsigned int 		d_lod2 					= 0;
unsigned int 		d_lod3 					= 0;
unsigned int 		d_total					= 0;
unsigned int		d_culled				= 0;
//->UNSIGNED
//=======================================================================================
//->POINTER
float 				view_mat[16];
float				proj_mat[16];
//<-POINTER
//=======================================================================================
//->STD
//-->VECTOR
vector<glm::vec3>	predef_cam_pos;
vector<glm::vec3>	predef_cam_dir;
vector<glm::vec3>	predef_cam_up;
vector<string>		str_mdp_csv;
//<--VECTOR
//=======================================================================================
//->CASIM
VboManager*         vbo_manager             = NULL;
GlslManager*        glsl_manager            = NULL;
FboManager*			fbo_manager				= NULL;
GlErrorManager*     err_manager             = NULL;
LogManager*         log_manager             = NULL;
Axes*               axes                    = NULL;
Camera*             camera1                 = NULL;
Camera*             camera2                 = NULL;
Camera*             camera                  = NULL;
SkyboxManager*		skybox_manager			= NULL;
CrowdManager*		crowd_manager			= NULL;
ObstacleManager*	obstacle_manager		= NULL;
SCENARIO_TYPE		scenario_type			= ST_CCM;
OBSTACLE_TYPE		obstacle_type			= OT_NONE;
CROWD_POSITION		crowd_position			= CPOS_NONE;
#ifdef DRAW_SCENARIO
	Scenario*		scenario				= NULL;
#endif
//<-CASIM
//=======================================================================================
//-->STRING
string				str_title							= string( "CASIM MULTIPLATFORM"				);
string				str_fps								= string( "FPS:        "					);
string				str_delta_time						= string( "DELTA TIME: "					);
string				str_culled							= string( "CULLED:     "					);
string				str_lod1							= string( "LOD1:       "					);
string				str_lod2							= string( "LOD2:       "					);
string				str_lod3							= string( "LOD3:       "					);
string				str_racing_qw						= string( "RQW:        "					);
string				str_scatter_gather					= string( "SG:         "					);
string				str_display_fbo						= string( "display_fbo" 					);
string				str_gl_Position						= string( "gl_Position"						);

string				str_instancing_textured				= string( "instancing_textured"				);
string				str_positions_textured				= string( "positions"						);
string				str_rotations_textured				= string( "rotations"						);
string				str_scales_textured					= string( "scales"							);
string				str_normal_textured					= string( "normal"							);
string				str_texCoord0_textured				= string( "texCoord0"						);
string				str_ViewMat4x4_textured				= string( "ViewMat4x4"						);

string				str_instancing_untextured			= string( "instancing_untextured"			);
string				str_positions_untextured			= string( "positions"						);
string				str_rotations_untextured			= string( "rotations"						);
string				str_scales_untextured				= string( "scales"							);
string				str_normal_untextured				= string( "normal"							);
string				str_ViewMat4x4_untextured			= string( "ViewMat4x4"						);

string				str_instancing_culled				= string( "instancing_culled"				);
string				str_instancing_culled_rigged		= string( "instancing_culled_rigged"		);
string				str_normalV							= string( "normalV"							);
string				str_dt								= string( "dt"								);
string				str_instancing_culled_rigged_shadow = string( "instancing_culled_rigged_shadow"	);
string				str_ProjMat4x4						= string( "ProjMat4x4"						);
string				str_ShadowMat4x4					= string( "ShadowMat4x4"					);
string				str_mdp_tex							= string( "mdp_tex"							);
string				str_mdp_fbo							= string( "mdp_fbo"							);
string				str_display_tex						= string( "display_tex"						);
string				str_display_depth_tex				= string( "display_depth_tex"				);
string				str_crowd_xml						= string( "Crowd.xml" 						);
string				str_pass_rect						= string( "pass_rect" 						);
string				str_policy_on						= string( "policy_on"						);
string				str_density_on						= string( "density_on" 						);
string				str_tang							= string( "tang" 							);
string				str_nearPlane						= string( "nearPlane" 						);
string				str_farPlane						= string( "farPlane" 						);
string				str_ratio							= string( "ratio" 							);
string				str_tile_side						= string( "tile_side" 						);
string				str_tile_height						= string( "tile_height" 					);
string				str_policy_width					= string( "policy_width" 					);
string				str_policy_height					= string( "policy_height" 					);
string				str_curr_dir						= string( ""								);
string				str_scenario_path					= string( ""								);
string				str_scenario_obj					= string( ""								);
string				str_skybox_front					= string( ""								);
string				str_skybox_back						= string( ""								);
string				str_skybox_left						= string( ""								);
string				str_skybox_right					= string( ""								);
string				str_skybox_top						= string( ""								);
string				str_skybox_bottom					= string( ""								);
string				str_scenario_speed					= string( ""								);
//<--STRING
//<-STD
//=======================================================================================
