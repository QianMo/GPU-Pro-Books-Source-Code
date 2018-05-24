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

//============================================================================================================
//
void blurSSAO( void )
{
	unsigned int ssao_width  = getWidth( "ssao_fbo" );
	unsigned int ssao_height = getHeight( "ssao_fbo" );

	bind_fbo( "ssao_fbo" );
	{
		proj_manager->setOrthoProjection( ssao_width, ssao_height );
		{
//->performing horizontal blur
			setTarget( "ssao_blurh_tex", true, true );
			pushActiveBind( "ssao_tex", 0 );
			{
				renderQuad( "gaussblur_horiz_rect",
							ssao_width,
							ssao_height,
							ssao_width,
							ssao_height				);
			}
			popActiveBind();
//->performing vertical blur
			setTarget( "ssao_tex", true, true );
			pushActiveBind( "ssao_blurh_tex", 0 );
			{
				renderQuad( "gaussblur_vert_rect",
							ssao_width,
							ssao_height,
							ssao_width,
							ssao_height				);
			}
			popActiveBind();
		}
		proj_manager->restoreProjection();
	}
	unbind_fbo( "ssao_fbo" );
}
//
//============================================================================================================
//
void renderToSSAO( void )
{
	unsigned int ssao_width  = getWidth( "ssao_fbo" );
	unsigned int ssao_height = getHeight( "ssao_fbo" );

	bind_fbo( "ssao_fbo" );
	{
		proj_manager->setOrthoProjection( ssao_width, ssao_height );
		{
			setTarget( "ssao_tex", true, true );
			pushActiveBind( "scene_depth_tex", 0 );
			{
				renderQuad( "ssao_rect",
							ssao_width,
							ssao_height,
							ssao_width,
							ssao_height );
			}
			popActiveBind();
		}
		proj_manager->restoreProjection();
	}
	unbind_fbo( "ssao_fbo" );
}
//
//============================================================================================================
//
void composeSSAO( void )
{
	unsigned int scene_width  = getWidth( "scene_fbo" );
	unsigned int scene_height = getHeight( "scene_fbo" );

	bind_fbo( "scene_fbo" );
	{
		proj_manager->setOrthoProjection( scene_width, scene_height );
		{
			setTarget( "composed_ssao_tex", true, true );
			pushActiveBind( "scene_tex", 0 );
			{
				pushActiveBind( "ssao_tex", 1 );
				{
					renderQuad( "ssao_compose_rect",
								scene_width,
								scene_height,
								scene_width,
								scene_height		);
				}
				popActiveBind();
			}
			popActiveBind();
		}
		proj_manager->restoreProjection();
	}
	unbind_fbo( "scene_fbo" );
}
//
//============================================================================================================
//
void blurShadows( void )
{
	unsigned int scene_width  = getWidth( "scene_fbo" );
	unsigned int scene_height = getHeight( "scene_fbo" );

	bind_fbo( "scene_fbo" );
	{
		proj_manager->setOrthoProjection( scene_width, scene_height );
		{
//->performing horizontal blur
			setTarget( "shadow_blurh_tex", true, true );
			pushActiveBind( "shadow_tex", 0 );
			{
				renderQuad( "gaussblur_horiz_rect",
							scene_width,
							scene_height,
							scene_width,
							scene_height			);
			}
			popActiveBind();
//->performing vertical blur
			setTarget( "shadow_tex", true, true );
			pushActiveBind( "shadow_blurh_tex", 0 );
			{
				renderQuad( "gaussblur_vert_rect",
							scene_width,
							scene_height,
							scene_width,
							scene_height			);
			}
			popActiveBind();
		}
		proj_manager->restoreProjection();
	}
	unbind_fbo( "scene_fbo" );
}
//
//============================================================================================================
//
void composeShadows( bool ssao_enabled )
{
	unsigned int scene_width  = getWidth( "scene_fbo" );
	unsigned int scene_height = getHeight( "scene_fbo" );

	bind_fbo( "scene_fbo" );
	{
		proj_manager->setOrthoProjection( scene_width, scene_height, false );
		{
			setTarget( "composed_shadow_tex", true, true );
			string input_texture;
			if( ssao_enabled )
			{
				input_texture = "composed_ssao_tex";
			}
			else
			{
				input_texture = "scene_tex";
			}
			pushActiveBind( input_texture, 0 );
			{
				pushActiveBind( "shadow_tex", 1 );
				{
					renderQuad( "shadow_compose_rect",
								scene_width,
								scene_height,
								scene_width,
								scene_height		);
				}
				popActiveBind();
			}
			popActiveBind();
		}
		proj_manager->restoreProjection();
	}
	unbind_fbo( "scene_fbo" );
}
//
//============================================================================================================
//
void downSampleBloom( void )
{
	unsigned int down_width   = getWidth ( "downsample_fbo" );
	unsigned int down_height  = getHeight( "downsample_fbo" );
	unsigned int scene_width  = getWidth ( "scene_fbo"		);
	unsigned int scene_height = getHeight( "scene_fbo"		);

	bind_fbo( "downsample_fbo" );
	{
		setTarget( "downsample_tex", true, true );
		proj_manager->setOrthoProjection( down_width, down_height );
		{
			pushActiveBind( "scene_tex", 0 );
			{
				renderQuad( "downsample_rect",
							down_width,
							down_height,
							scene_width,
							scene_height	);
			}
			popActiveBind();
		}
		proj_manager->restoreProjection();
	}
	unbind_fbo( "downsample_fbo" );
}
//
//============================================================================================================
//
void renderToBloom( void )
{
	unsigned int down_width  = getWidth ( "downsample_fbo" );
	unsigned int down_height = getHeight( "downsample_fbo" );

	bind_fbo( "downsample_fbo" );
	{
		proj_manager->setOrthoProjection( down_width, down_height );
		{
//->performing horizontal blur
			setTarget( "downsample_bloom_blurh_tex", true, true );
			pushActiveBind( "downsample_tex", 0 );
			{
				renderQuad( "gaussblur_half_horiz_rect",
							down_width,
							down_height,
							down_width,
							down_height				);
			}
			popActiveBind();
//->performing vertical blur
			setTarget( "downsample_final_blur_tex", true, true );
			pushActiveBind( "downsample_bloom_blurh_tex", 0 );
			{
				renderQuad( "gaussblur_half_vert_rect",
							down_width,
							down_height,
							down_width,
							down_height				);
			}
			popActiveBind();
		}
		proj_manager->restoreProjection();
	}
	unbind_fbo( "downsample_fbo" );
}
//
//============================================================================================================
//
void composeBloom( bool ssao_enabled, bool shadow_enabled )
{
	unsigned int scene_width  = getWidth ( "scene_fbo" );
	unsigned int scene_height = getHeight( "scene_fbo" );

	bind_fbo( "scene_fbo" );
	{
		proj_manager->setOrthoProjection( scene_width, scene_height );
		{
			setTarget( "composed_bloom_tex", true, true );
			string input_texture;
			if( shadow_enabled )
			{
				input_texture = "composed_shadow_tex";
			}
			else if( ssao_enabled )
			{
				input_texture = "composed_ssao_tex";
			}
			else
			{
				input_texture = "scene_tex";
			}
			pushActiveBind( input_texture, 0 );
			{
				pushActiveBind( "downsample_final_blur_tex", 1 );
				{
					renderQuad( "bloom_compose_rect",
								scene_width,
								scene_height,
								scene_width,
								scene_height		);
				}
				popActiveBind();
			}
			popActiveBind();
		}
		proj_manager->restoreProjection();
	}
	unbind_fbo( "scene_fbo" );
}
//
//============================================================================================================
