// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

#ifndef __ARG_PARSER_H__
#define __ARG_PARSER_H__

#include <glibmm.h>
#include <iostream>
#ifdef WINDOWS
#include "WindowsUtil.h"
#elif defined (LINUX)
#include <DataTypes.h>
#endif

class ArgumentsParser : public Glib::OptionGroup
{
private:
	Glib::OptionContext		m_OptionContext; // context of glib
	GKeyFile*				m_pKeyFile; // ini file to load
	GError*					m_pError; // error pointer

	// Global
	int						m_iNumThreads; // number of threads used by the cpu
	int						m_iIterations; // number of frames rendered before quitting
	
	// Camera-Model variables
	Glib::ustring			m_sModel; // model to load
	float					m_fSpeed; // camera speed
	
	// Reflections
	int						m_iNumReflections; // number of inital reflections
	bool					m_bIsReflective; // reflections are on?
	bool					m_bIsMultiplicative; // are reflections multiplicative?
	
	// Acceleration Structures
	Glib::ustring			m_sAccelerationStruct; // name of the acceleration structure
	Glib::ustring			m_sBVHSplit; // split algorithm for the BVH
	int						m_iMaxPrimsNode; // maximum number of primitives per node
	int						m_iLBVHDepth; // LBVH depth of the tree

	// Screen
	int						m_iTextureWidth; // width of the render target texture
	int						m_iTextureHeight; // height of the render target texture
	int						m_iScreenMultiplier; // multiplier for textures

	// GPU execution
	int						m_iGroupSizeX; // number of groups in x-dim to dispatch in GPU
	int						m_iGroupSizeY; // number of groups in x-dim to dispatch in GPU
	int						m_iGroupSizeZ; // number of groups in x-dim to dispatch in GPU


public:
	ArgumentsParser();
	~ArgumentsParser();

	// Glib functions
	virtual bool			on_pre_parse(Glib::OptionContext& context, Glib::OptionGroup& group);
	virtual bool			on_post_parse(Glib::OptionContext& context, Glib::OptionGroup& group);
	virtual void			on_error(Glib::OptionContext& context, Glib::OptionGroup& group);

	// Parser functions
	int						ParseData();
	int						LoadConfigurationFromFile(const char* sFile);
	void					ShowConfiguration(unsigned int uiProcesses);
	
	// Getters
	unsigned int			GetNumThreads() { return static_cast<unsigned int>(m_iNumThreads); }
	unsigned int			GetNumReflections() { return static_cast<unsigned int>(m_iNumReflections); }
	unsigned int			GetMaxPrimsInNode() { return static_cast<unsigned int>(m_iMaxPrimsNode); }
	unsigned int			GetTextureWidth() { return static_cast<unsigned int>(m_iTextureWidth); }
	unsigned int			GetTextureHeight() { return static_cast<unsigned int>(m_iTextureHeight); }
	unsigned int			GetScreenMultiplier() { return static_cast<unsigned int>(m_iScreenMultiplier); }
	unsigned int			GetGroupSizeX() { return static_cast<unsigned int>(m_iGroupSizeX); }
	unsigned int			GetGroupSizeY() { return static_cast<unsigned int>(m_iGroupSizeY); }
	unsigned int			GetGroupSizeZ() { return static_cast<unsigned int>(m_iGroupSizeZ); }
	unsigned int			GetNumIterations() { return static_cast<unsigned int>(m_iIterations); }
	unsigned int			GetLBVHDepth() { return static_cast<unsigned int>(m_iLBVHDepth); }
	float					GetSpeed() { return m_fSpeed; }
	const char*				GetModelName() { return m_sModel.c_str(); }
	const char*				GetBVHSplit() { return m_sBVHSplit.c_str(); }
	const char*				GetAccelerationStructure() { return m_sAccelerationStruct.c_str(); }
	bool					IsReflective() { return m_bIsReflective; }
	bool					IsMultiplicative() { return m_bIsMultiplicative; }
};

#endif //__ARG_PARSER_H__
