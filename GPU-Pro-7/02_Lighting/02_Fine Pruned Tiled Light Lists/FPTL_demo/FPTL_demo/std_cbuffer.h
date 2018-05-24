#ifndef __STDCBUFFER_H__
#define __STDCBUFFER_H__

#include "shader_base.h"


unistruct cbMeshInstance
{
	Mat44	g_mWorldViewProjection;
	Mat44	g_mScrToView;
	Mat44	g_mLocToView;
	
	Vec3	g_vCamPos;
	int		g_iMode;

	int		g_iWidth;
	int		g_iHeight;
};

#endif