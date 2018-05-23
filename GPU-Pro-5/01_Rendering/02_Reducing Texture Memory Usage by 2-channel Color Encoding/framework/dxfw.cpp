// ---------------- Library -- generated on 4.9.2013  0:33 ----------------



// ******************************** device.cpp ********************************

// ---- #include "dxfw.h"
// ---> including dxfw.h

#pragma once


#define WINVER 0x0501
#define _WIN32_WINNT 0x0501


#include <d3d9.h>
#include <d3dx9.h>

#include <assert.h>

#include <vector>
#include <string>
#include <algorithm>

#include "base.h"


/** \file dxfw.h
 * \brief Main DxFw header file.
 *
 * This is the header file of DxFW library.
 */



#if defined(_DEBUG) && defined(USE_DEBUG_D3D)
#pragma comment(lib, "d3d9.lib")
#pragma comment(lib, "d3dx9d.lib")
#else
#pragma comment(lib, "d3d9.lib")
#pragma comment(lib, "d3dx9.lib")
#endif

#pragma comment(lib, "winmm.lib")

#ifndef _DEBUG
// ---- #pragma comment(lib, "dxfw.lib")
#else
// ---- #pragma comment(lib, "dxfw_d.lib")
#endif


#define ATTR_DECL(usage,id)		(((usage)<<16) | (id))

enum tMeshAttrib {
	ATTR_NONE			= ATTR_DECL(D3DX_DEFAULT, 0),
	ATTR_POSITION		= ATTR_DECL(D3DDECLUSAGE_POSITION, 0),
	ATTR_NORMAL			= ATTR_DECL(D3DDECLUSAGE_NORMAL, 0),
	ATTR_PSIZE			= ATTR_DECL(D3DDECLUSAGE_PSIZE, 0),
	ATTR_TEXCOORD_0		= ATTR_DECL(D3DDECLUSAGE_TEXCOORD, 0),
	ATTR_TEXCOORD_1		= ATTR_DECL(D3DDECLUSAGE_TEXCOORD, 1),
	ATTR_TEXCOORD_2		= ATTR_DECL(D3DDECLUSAGE_TEXCOORD, 2),
	ATTR_TEXCOORD_3		= ATTR_DECL(D3DDECLUSAGE_TEXCOORD, 3),
	ATTR_TEXCOORD_4		= ATTR_DECL(D3DDECLUSAGE_TEXCOORD, 4),
	ATTR_TEXCOORD_5		= ATTR_DECL(D3DDECLUSAGE_TEXCOORD, 5),
	ATTR_TEXCOORD_6		= ATTR_DECL(D3DDECLUSAGE_TEXCOORD, 6),
	ATTR_TEXCOORD_7		= ATTR_DECL(D3DDECLUSAGE_TEXCOORD, 7),
};

#define ATTR_GET_USAGE(attr)	(int(attr)>>16)
#define ATTR_GET_ID(attr)		((attr)&0xFFFF)
#define ATTR_IS_PRESENT(attr)	((attr)!=ATTR_NONE)

#define TEXF_MIPMAPS			(1<<0)
#define TEXF_AUTO_MIPMAPS		(1<<1)
#define RT_PRIORITY(level)		((level)<<24)

#define GET_RT_PRIORITY(f)		(((f)>>24)&0xF)

#define FXF_NO_ERROR_POPUPS			(1<<0)
#define FXF_FLOW_CONTROL_MEDIUM		(1<<1)
#define FXF_FLOW_CONTROL_HIGH		(1<<2)
#define FXF_PARTIAL_PRECISION		(1<<3)
#define FXF_OPTIMIZATION_0			(1<<4)
#define FXF_OPTIMIZATION_1			(1<<5)
#define FXF_OPTIMIZATION_2			(1<<6)
#define FXF_OPTIMIZATION_3			(1<<7)
#define FXF_LEGACY                  (1<<8)


// SetRState flags
//   blend_dest  blend_src         alpha_ref              write disable             blend_op  cull
// |   d d d d    s s s s   |  a a a a a a a a  | - - - -   a  r  g  b  | -   o o o     c c  zwrite zenable |
#define RSF_NO_ZENABLE              (1<<0)
#define RSF_NO_ZWRITE               (1<<1)
#define RSF_CULL(c)                 (((c)-1)<<2)
#define RSF_CULL_CW                 RSF_CULL(D3DCULL_CW)
#define RSF_CULL_CCW                RSF_CULL(D3DCULL_CCW)
#define RSF_ALPHA_TEST(v)           (((v)&0xFF)<<16)
#define RSF_ALPHA_BLEND(s,d)        ( ((((s)-1)&0xF)<<24) | ((((d)-1)&0xF)<<28) | 0x00000010)
#define RSF_ALPHA_BLEND_OP(s,d,op)  ( ((((s)-1)&0xF)<<24) | ((((d)-1)&0xF)<<28) | (((op)&0x7)<<4) )
#define RSF_WRITE_RGBA              0
#define RSF_WRITE_RGB               (0x8<<8)
#define RSF_WRITE_A                 (0x7<<8)
#define RSF_WRITE_MASK(x)           (((~(x))&0xF)<<8)

#define RSF_BLEND_ALPHA             RSF_ALPHA_BLEND(D3DBLEND_SRCALPHA,D3DBLEND_INVSRCALPHA)
#define RSF_BLEND_PREMUL_ALPHA      RSF_ALPHA_BLEND(D3DBLEND_ONE,D3DBLEND_INVSRCALPHA)
#define RSF_BLEND_ADD               RSF_ALPHA_BLEND(D3DBLEND_INVDESTCOLOR,D3DBLEND_ONE)
#define RSF_BLEND_ADD_RAW           RSF_ALPHA_BLEND(D3DBLEND_ONE,D3DBLEND_ONE)
#define RSF_BLEND_MUL               RSF_ALPHA_BLEND(D3DBLEND_ZERO,D3DBLEND_SRCCOLOR)
#define RSF_BLEND_MUL_X2            RSF_ALPHA_BLEND(D3DBLEND_DESTCOLOR,D3DBLEND_SRCCOLOR)
#define RSF_BLEND_SUB               RSF_ALPHA_BLEND_OP(D3DBLEND_ONE,D3DBLEND_ONE,D3DBLENDOP_REVSUBTRACT)
#define RSF_BLEND_SUB_SMOOTH        RSF_ALPHA_BLEND_OP(D3DBLEND_DESTCOLOR,D3DBLEND_ONE,D3DBLENDOP_REVSUBTRACT)




// SetSampler texture flags
#define TEXF_POINT				0
#define TEXF_LINEAR				1
#define TEXF_MIPMAP				2
#define TEXF_MIPMAP_LINEAR		3
#define TEXF_ANISO(n)			((n)+2)	// n>=2

#define TEXA_WRAP			(D3DTADDRESS_WRAP<<5)
#define TEXA_CLAMP			(D3DTADDRESS_CLAMP<<5)
#define TEXA_BORDER			(D3DTADDRESS_BORDER<<5)

#define TEXB_BLACK          0
#define TEXB_WHITE          (1<<8)



class Device;



/** \brief Base class for device resources.
 *
 * This class is a base class of device resources.
 * It provides callbacks for device events, in response to which
 * the resource may want to perform certain tasks.
 *
 * Each resource is automatically tracked by resource manager.
 * IDevResource constructor and destructor automatically registers and unregisters
 * the resource.
 */
class IDevResource {
public:
	//! Constructor.
	IDevResource();
	//! Destructor.
	virtual ~IDevResource();

protected:
	friend class DeviceResourceManager;

	//! Callback called before DirectX device is reset.
	/** Resource should release DirectX resources required to be released before device reset.*/
	virtual void OnBeforeReset() = 0;
	//! Callback called after DirectX device reset is completed.
	virtual void OnAfterReset() = 0;
	//! Callback called after DirectX device is created.
	/** Mainly used for preloading resources defined as global variables.*/
	virtual void OnPreload() { };
	//! Returns relative priority of handling order (default=0).
	virtual int  GetPriority() { return 0; }
	//! Called periodically when autoreload option is enabled.
	virtual void OnReloadTick() { };
};


/** \brief Device texture class.
 *
 * This class is a wrapper for DirectX textures.
 */
class DevTexture : public IDevResource {
public:
	//! Constructor.
	DevTexture();
	//! Constructor for global variables (resource loads after device creation).
	/** \param _path path to texture being loaded*/
	DevTexture(const char *_path);
	//! Destructor.
	~DevTexture();

	//! Releases associated DirectX texture.
	void Unload();
	//! Loads texture from file
	/** \param path path to texture being loaded
	  * \return true on success, false on failure*/
	bool Load(const char *path);
	//! Loads cube texture from file
	/** \param path path to texture being loaded
	  * \return true on success, false on failure*/
	bool LoadCube(const char *path);
	bool LoadRaw2D(int format,int w,int h,const void *data,bool use_mipmaps);
	bool CreateEmpty2D(int format,int w,int h,bool create_mipmaps,bool autogen_mipmaps);
	bool BuildLookup2D(int w,int h,DWORD (*fn)(float,float),bool use_mipmaps);
//	bool BuildLookup3D(int sx,int sy,int sz,DWORD (*fn)(float,float,float),bool use_mipmaps);
	bool BuildLookupCube(int size,DWORD (*fn)(float,float,float),bool use_mipmaps);

	bool GetRawData(int &width,int &height,std::vector<DWORD> &data);

	base::vec2 GetSize2D();

	IDirect3DBaseTexture9 *GetTexture() { return tex; }
	operator IDirect3DBaseTexture9*() { return tex; }

protected:
	IDirect3DBaseTexture9	*tex;
	std::string				preload_path;

	virtual void OnBeforeReset();
	virtual void OnAfterReset();
	virtual void OnPreload();
};


class DevRenderTarget : public IDevResource {
public:
	DevRenderTarget();
	DevRenderTarget(int _format,int _width,int _height,int _denominator,int _flags = 0);
	~DevRenderTarget();

	IDirect3DTexture9 *GetTexture() { return rt; }
	IDirect3DSurface9 *GetSurface();
	
	void SetParameters(int _format,int _width,int _height,int _denominator,int _flags = 0);
	void GetCurrentSize(int &sx,int &sy);
	base::vec2 GetCurrentSizeV();

	bool Save(const char *path);

	operator IDirect3DBaseTexture9*() { return rt; }

protected:
	IDirect3DTexture9	*rt;
	int					format;
	int					width;
	int					height;
	int					denominator;
	int					flags;

	virtual void OnBeforeReset();
	virtual void OnAfterReset();
	virtual int  GetPriority() { return GET_RT_PRIORITY(flags); }
};


class DevCubeRenderTarget : public IDevResource {
public:
	DevCubeRenderTarget(int format,int _size);
	~DevCubeRenderTarget();

	IDirect3DCubeTexture9 *GetTexture() { return rt; }
	IDirect3DSurface9 *GetSurface(int face);
	
	void SetParameters(int _format,int _size);
	int GetSize() { return size; }

	operator IDirect3DBaseTexture9*() { return rt; }

protected:
	IDirect3DCubeTexture9	*rt;
	int						format;
	int						size;

	virtual void OnBeforeReset();
	virtual void OnAfterReset();
};

class DevDepthStencilSurface : public IDevResource {
public:
	DevDepthStencilSurface(int format,int _width,int _height,int _denominator);
	~DevDepthStencilSurface();

	IDirect3DSurface9 *GetSurface();
	
	void GetCurrentSize(int &sx,int &sy);

	operator IDirect3DSurface9*() { return surf; }

protected:
	IDirect3DSurface9	*surf;
	int					format;
	int					width;
	int					height;
	int					denominator;

	virtual void OnBeforeReset();
	virtual void OnAfterReset();
};



class DevMesh : public IDevResource {
public:
	enum {	GDE_IN_PLACE		= (1<<0),
			GDE_ADD_ONE_SIDED	= (1<<1),
			GDE_ADD_FLATS		= (1<<2),
	};

	struct Range {
		DWORD	attrib;
		int		face_start;
		int		face_count;
		int		vtx_start;
		int		vtx_count;
	};

	ID3DXMesh	*mesh;

	DevMesh();
	DevMesh(const char *_path);
	~DevMesh();

	bool	Load(const char *path);
	bool	LoadVBIB(const void *vdata,int vcount,int vsize,int FVF,int *idata,int icount,bool optimize=true,DWORD *attr=NULL);
	bool	Save(const char *path);

	bool	LoadDXMesh(ID3DXMesh *m) { Clear(true); mesh=m; return true; }
    bool    LoadCube(float w,float h,float d);
    bool    LoadSphere(float r,int slices,int stacks);
    bool    LoadCylinder(float r1,float r2,float len,int slices,int stacks);
    bool    LoadTorus(float rin,float rout,int sides,int rings);
    bool    LoadPolygon(float len,int sides);

	void	Clear(bool all);
	void	GenerateAdjacency(float epsilon);
	void	GenerateSoftAdjacency(float epsilon,float normal_dot,bool normal_dir);
	bool	ReadBack();
	bool	Upload();
	void	ApplyMatrixInMemory(D3DXMATRIX *mtx);
	void	FlipTrisInMemory();
	bool	ComputeTangentFrame(tMeshAttrib tex_attr,tMeshAttrib normal_attr,
								tMeshAttrib tangent_attr,tMeshAttrib binormal_attr,
								float dot_normal,float dot_tangent,float eps_singular,int options);

	void	GenerateDegenerateEdges(int flags,tMeshAttrib normal_attr,DevMesh *out);
	bool	ReorderVertexFields(int FVF,tMeshAttrib attrf_map[][2]);
	bool	UnwrapUV(float max_stretch,int tex_w,int tex_h,float gutter,int tex_id,
					 float normal_dot=-1,bool normal_dir=false);
	bool	CleanMesh();
	bool	Optimize();
	bool	Simplify(int size,bool count_faces);

	int		GetVertexDataSize() { return GetVertexStride()*GetVertexCount(); }
	int		GetVertexStride();
	int		GetVertexCount();
	bool	CopyVertexData(void *buffer);
	int		GetIndexCount();
	bool	CopyIndexData(int *buffer);
	int		GetRangeCount();
	bool	CopyRangeData(Range *buffer);

	void	DrawSection(int id);
	void	DrawRange(const Range &r);
	void	DrawTriangleRange(int first_tri,int num_tris);

	virtual void OnBeforeReset();
	virtual void OnAfterReset();
	virtual void OnPreload();

//private:
	DWORD	*adjacency;
	void	*vdata;
	int		*idata;
	DWORD   *adata;
	int		vsize;
	int		vcount;
	int		icount;

private:
	std::string	preload_path;

//	void	MakeDeclFromFVFCode(int FVF,int &decl,int &id);
};


class DevFont : public IDevResource {
public:

	DevFont();
	DevFont(const char *facename,int height,int width,bool bold,bool italic);
	~DevFont();

	bool Create(const char *facename,int height,int width,bool bold,bool italic);

	ID3DXFont *GetFont() { return font; }
	ID3DXFont *operator ->() { return font; }

protected:
	ID3DXFont		*font;
	bool			do_preload;
	std::string		p_facename;
	int				p_height;
	int				p_width;
	bool			p_bold;
	bool			p_italic;

	virtual void OnBeforeReset();
	virtual void OnAfterReset();

	virtual void OnPreload()
	{
		if(do_preload)
		{
			Create(p_facename.c_str(),p_height,p_width,p_bold,p_italic);
			do_preload = false;
		}
	}
};


class DevEffect : public IDevResource {
public:
	typedef void DefGenerator(std::vector<std::string> &defs);

	DevEffect() : fx(NULL), pool(NULL), do_preload(false), pass(-2), def_generator(NULL),
				  flags(0), compile_flags(D3DXSHADER_AVOID_FLOW_CONTROL) {}
	DevEffect(const char *path,DefGenerator *dg=NULL,int _flags=0);

	~DevEffect()
	{
		if(pool) pool->Release();
		for(int i=0;i<(int)fx_list.size();i++)
			fx_list[i]->Release();
		for(int i=0;i<(int)textures.size();i++)
			delete textures[i];
	}

	bool		Load(const char *path,DefGenerator *dg=NULL);
	const char *GetTechniqueName(int id);
	const char *GetTechniqueInfo(const char *name,const char *param);

	bool		SelectVersionStr(const char *version);
	bool		SelectVersion(int version);
	int			GetVersionIndex(const char *version);

	bool		StartTechnique(const char *name);
	bool		StartPass();

	void		SetFloat (const char *name,float v) { fx->SetFloat(name,v); }
	void		SetFloat2(const char *name,const base::vec2 &v) { fx->SetFloatArray(name,&v.x,2); }
	void		SetFloat3(const char *name,const base::vec3 &v) { fx->SetFloatArray(name,&v.x,3); }
	void		SetFloat4(const char *name,const base::vec4 &v) { fx->SetFloatArray(name,&v.x,4); }

	ID3DXEffect *GetEffect() { assert(fx!=NULL); return fx; }
	ID3DXEffect *operator ->() { assert(fx!=NULL); return fx; }

	const char	*GetLastError() { return last_error.c_str(); }


protected:
	ID3DXEffectPool				*pool;
	ID3DXEffect					*fx;
	std::vector<ID3DXEffect*>	fx_list;
	std::vector<DevTexture*>	textures;
	int							pass;
	int							n_passes;
	bool						do_preload;
	std::string					p_path;
	DefGenerator				*def_generator;
	std::map<std::string,int>	version_index;
	int							flags;
	DWORD						compile_flags;
	std::string					last_error;
	unsigned long long			file_time;

	virtual void OnBeforeReset();
	virtual void OnAfterReset();

	virtual void OnReloadTick()
	{
		if(p_path.size()>0 && base::GetFileTime(p_path.c_str())>file_time)
			Load(p_path.c_str(),def_generator);
	}

	virtual void OnPreload()
	{
		if(p_path.size()>0)
			Load(p_path.c_str(),def_generator);
	}

	void ClearEffect();
	bool ScrambleLoad(const char *path,std::vector<char> &data);
	void BuildSkipList(const char *d,std::string &skip_list);
	void ShatterDefsList(char *s,std::vector<D3DXMACRO> &macros);
	bool GetPrecompiledBinary( std::vector<char> &data, const char *id, const char *path, const char *bpath,
							   std::vector<D3DXMACRO> &macros, byte hash[16], std::vector<byte> &bin );
	bool CompileVersion( const char *id, const char *path, std::vector<byte> &bin,
						 std::vector<D3DXMACRO> &macros, const char *skip_list );

};


class DevShaderSet : public IDevResource {
public:
	typedef void DefGenerator(std::vector<std::string> &defs);

	DevShaderSet(const char *path,DefGenerator *dg,const char *v="2_0");
	~DevShaderSet();

	bool		Load(const char *path,DefGenerator *dg=NULL);

	bool		BindVersionStr(const char *version);
	bool		BindVersion(int version);
	void		Unbind();
	int			GetVersionIndex(const char *version);

protected:
	struct ShaderSet {
		DWORD					name_ofs;
		DWORD					vcode_ofs;
		DWORD					pcode_ofs;
		IDirect3DVertexShader9	*vshader;
		IDirect3DPixelShader9	*pshader;
	};
	
	std::vector<char>			source;
	std::vector<std::string>	defs;
	std::vector<D3DXMACRO>		macro_ptrs;
	std::vector<DWORD>			macro_start;
	std::string					original_path;
	std::string					final_save_path;

	std::vector<byte>			raw_file;
	std::vector<ShaderSet>		versions;
	std::string					p_path;
	DefGenerator				*def_generator;
	const char					*version;
	
	std::map<std::string,int>	vmap;


	virtual void OnBeforeReset();
	virtual void OnAfterReset();

	virtual void OnPreload()
	{
		if(p_path.size()>0)
		{
			Load(p_path.c_str(),def_generator);
			p_path.clear();
		}
	}

	void ClearAll();
	void ReleaseAll();
	bool ScrambleLoad(const char *path,std::vector<char> &data);
	void ShatterDefsList(char *s,std::vector<D3DXMACRO> &macros);
	void CreateShaderCache( std::vector<std::string> &defs, byte hash[16] );
	bool LoadShaderCache( const char *path, byte hash[16] );
	bool SaveShaderCache( const char *path );
	void CompileVersion( int ver, bool create_shaders );

};





class IDevMsgHandler {
public:
	virtual LRESULT MsgProc(Device *dev,HWND hWnd,UINT msg,WPARAM wParam,LPARAM lParam,bool &pass_forward) = 0;
};


class DeviceResourceManager {
public:

private:
	// singleton handling
	struct DRMHandler {
		DRMHandler() { DeviceResourceManager::GetInstance(); }
		~DRMHandler() {
			if(DeviceResourceManager::singleton)
				delete DeviceResourceManager::singleton;
			DeviceResourceManager::singleton = NULL;
		}
	};

	static DeviceResourceManager	*singleton;
	static DRMHandler				handler;

	static DeviceResourceManager *GetInstance()
	{
		if(!singleton) singleton = new DeviceResourceManager();
		return singleton;
	}

	static DeviceResourceManager *PeekInstance()
	{
		return singleton;
	}

	// interface
	friend class IDevResource;
	friend class Device;

	std::vector<IDevResource*>	reslist;


	void Register(IDevResource *res);
	void Unregister(IDevResource *res);
	void SendOnBeforeReset();
	void SendOnAfterReset();
	void SendOnPreload();
	void SendOnReloadTick();
	void CompactTable();
};

class RawMouse {
public:
    int     xp;
    int     yp;
    int     dx;
    int     dy;
    int     dz;
    int     bt_down;            // flags
    int     bt_click;
    int     bt_2click;
    float   dbclick_timeout;

    base::vec2 GetPos()     { return base::vec2(float(xp),float(yp)); }
    base::vec2 GetDelta()   { base::vec2 d = base::vec2(float(dx),float(dy)); dx=dy=0; return d; }

    void ClampPos(int x1,int y1,int x2,int y2) { xp=max(x1,min(x2,xp)); yp=max(y1,min(y2,yp)); }

private:
    friend class Device;

    HANDLE  handle;
    double  press_time[5];      // time of last down event (via GetElapsedTime)

    void Clear() { memset(this,0,sizeof(*this)); dbclick_timeout=.3f; }
};

class Device {
public:
    
    Device();
	~Device();

	// Direct3D
	IDirect3DDevice9 *operator ->()	{ return dev; }
	IDirect3DDevice9 *GetDevice()	{ return dev; }

	// device & window management
	bool BeginScene();
	void EndScene();
	bool Sync();
	void SetResolution(int width,int height,bool fullscreen,int msaa=0);
	void SetAppName(const char *name);
	void SetMinZStencilSize(int width,int height);
	bool PumpMessages();
	bool Init(bool soft_vp=false);
	void Shutdown();
	void ForceReset() { d3d_need_reset = true; }
	void SetReloadTimer(int msec) { time_reload = msec/1000.f; }

	bool GetIsReady()	{ return (dev!=NULL) && (d3d_hWnd!=NULL); }
	HWND GetHWnd()		{ return d3d_hWnd; }

	void  GetScreenSize(int &sx,int &sy)	{ sx = d3d_screen_w; sy = d3d_screen_h; }
	base::vec2  GetScreenSizeV()			{ return base::vec2(float(d3d_screen_w),float(d3d_screen_h)); }
	int   GetSizeX()						{ return d3d_screen_w; }
	int   GetSizeY()						{ return d3d_screen_h; }
	float GetAspectRatio()					{ return float(d3d_screen_w)/float(d3d_screen_h); }

	void RegisterMessageHandler(IDevMsgHandler *dmh);
	void UnregisterMessageHandler(IDevMsgHandler *dmh);

	bool MainLoop();

	// render states
	void SetDefaultRenderStates();
    void SetRState(int mode);
	void SetSampler(int id,int flags);

	// rendertargets
	void SetRenderTarget(int id,DevRenderTarget *rt);
	void SetCubeRenderTarget(int id,DevCubeRenderTarget *rt,int face);
	void SetDepthStencil(DevDepthStencilSurface *ds);

	// buffer copying
	void StretchRect(DevRenderTarget *src,const RECT *srect,
					 DevRenderTarget *dst,const RECT *drect,
					 D3DTEXTUREFILTERTYPE filter);

	// 2D drawing
	void DrawScreenQuad(float x1,float y1,float x2,float y2,float z=0);
	void DrawScreenQuadTC(float x1,float y1,float x2,float y2,float *coord_rects,int n_coords,float z=0);
	void DrawScreenQuadTCI(float x1,float y1,float x2,float y2,float *tc00,float *tc10,float *tc01,float *tc11,
							int n_interp,int tc_fvf,float z=0);
	void DrawScreenQuadTCIR(float x1,float y1,float x2,float y2,float *rects,int n_interp,int tc_fvf,float z=0);
	void DrawScreenQuadVS(float x1,float y1,float x2,float y2,float z=0);

	void DrawSkyBoxQuad(const base::vec3 &ypr,float fov);

	// text drawing
	void Print(DevFont *fnt,int xp,int yp,int color,const char *text);
	void PrintF(DevFont *fnt,int xp,int yp,int color,const char *fmt, ...);
	void AlignPrint(DevFont *fnt,int xp,int yp,int align,int color,const char *text);
	void AlignPrintF(DevFont *fnt,int xp,int yp,int align,int color,const char *fmt, ...);
	void DPrintF(const char *fmt, ...);

	// helpers
	int *GetQuadsIndices(int n_quads);

	// mouse
	void SetMouseCapture(bool capture);
	void GetMousePos(int &mx,int &my)		{ mx = mouse_x; my = mouse_y; }
	base::vec2 GetMousePosV()				{ return base::vec2(float(mouse_x),float(mouse_y)); }
	void GetMouseDelta(int &dx,int &dy)		{ dx = mouse_dx; dy = mouse_dy; mouse_dx = 0; mouse_dy = 0; }
	base::vec2 GetMouseDelta()				{ base::vec2 out((float)mouse_dx,(float)mouse_dy);
											  mouse_dx = 0; mouse_dy = 0; return out; }
	base::vec2 PeekMouseDelta()				{ return base::vec2((float)mouse_dx,(float)mouse_dy); }

	float	   GetMouseDeltaZ()				{ float out = mouse_dz; mouse_dz = 0; return out; }
	float	   PeekMouseDeltaZ()			{ return mouse_dz; }

    // raw mice
    RawMouse &GetMiceData(int id)           { return mouse_data[id]; }
    int       GetMiceCount()                { return mouse_data.size(); }

    

	// keyboard
	int  ReadKey() { if(read_keys.size()<=0) return 0; int k=read_keys[0]; read_keys.erase(read_keys.begin()); return k; }
    bool GetKeyStroke(int vk);
	bool GetKeyState(int vk) { return has_focus && (::GetKeyState(vk) < 0); }

	// time
	float	GetTimeDelta()		{ return time_delta; }
	double	GetElapsedTime()	{ return time_elapsed; }


	// camera functions
	base::vec3	debug_cam_pos;
	base::vec3	debug_cam_ypr;

	void BuildProjectionMatrix(D3DXMATRIX *out,float vport[4],int rt_size_x,int rt_size_y,
							   const base::vec3 &ax,const base::vec3 &ay,const base::vec3 &az,float zn,float zf);

	void BuildCubemapProjectionMatrix(D3DXMATRIX *mtx,int face,int size,float zn,float zf,float *vport=NULL);

	void BuildCameraViewProjMatrix(D3DXMATRIX *mtx,const base::vec3 &pos,const base::vec3 &ypr,float fov,
									float aspect,float zn,float zf,bool m_view=true,bool m_proj=true);
	void BuildCameraVectors(const base::vec3 &ypr,base::vec3 *front,base::vec3 *right,base::vec3 *up);

	void TickFPSCamera(base::vec3 &pos,base::vec3 &ypr,float move_speed,float sens,bool flat=true);
	
	void RunDebugCamera(float speed,float sens=0.2f,float zn=0.01f,float fov=70.f,bool flat=true);


	// format information
	typedef void	fn_convert_u_t(DWORD rgba,void *out);
	typedef void	fn_convert_f_t(const float *rgba,void *out);

	static int				GetSurfaceSize(D3DFORMAT format,int w,int h);
	static fn_convert_u_t	*GetFormatConversionFunctionU(D3DFORMAT format);
	static fn_convert_f_t	*GetFormatConversionFunctionF(D3DFORMAT format);
	static bool				IsFormatDepthStencil(D3DFORMAT format);

private:
	friend static LRESULT WINAPI _DeviceMsgProc( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam );
	friend class IDevResource;

	IDirect3D9						*d3d;
	IDirect3DDevice9				*dev;
	HWND							d3d_hWnd;
	WNDCLASSEX						d3d_wc;
	D3DPRESENT_PARAMETERS			d3dpp;
	bool							has_focus;
	IDirect3DSurface9				*custom_zstencil;
	IDirect3DSurface9				*custom_zstencil_rt;
	IDirect3DSurface9				*block_surface[2];
	int								block_id;
	DevFont							default_font;
	std::vector<int>				quad_idx;
	
	std::string						d3d_app_name;
	bool							d3d_need_reset;
	bool							d3d_sizing;
	int								d3d_screen_w;
	int								d3d_screen_h;
	int								d3d_msaa;
	bool							d3d_windowed;
	int								d3d_min_zstencil_w;
	int								d3d_min_zstencil_h;
	bool							set_resolution_first;
	bool							quited;

	std::vector<IDevMsgHandler*>	msg_handlers;

	std::vector<int>				read_keys;
	int								mouse_x;
	int								mouse_y;
	int								mouse_dx;
	int								mouse_dy;
	float							mouse_dz;
	bool							mouse_capture;
    std::vector<RawMouse>           mouse_data;

	DWORD							time_prev;
	float							time_delta;
	double							time_elapsed;
	bool							time_first_frame;
	float							time_reload;
	double							time_last_reload;

	std::vector<std::string>		dprintf_buff;


	D3DPRESENT_PARAMETERS	*GetPresentParameters();
	LRESULT					MsgProc( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam );
	void					CreateInternalResources();
	void					ReleaseInternalResources();


	void SendOnBeforeReset()	{ DeviceResourceManager::GetInstance()->SendOnBeforeReset(); }
	void SendOnAfterReset()		{ DeviceResourceManager::GetInstance()->SendOnAfterReset(); }
	void PreloadResources()		{ DeviceResourceManager::GetInstance()->SendOnPreload(); }
	void SendOnReloadTick()		{ DeviceResourceManager::GetInstance()->SendOnReloadTick(); }

};

extern Device Dev;



class DevCanvas;


class DevTileSet : public IDevResource {
public:
	struct TileInfo {
		base::vec2	uvmin;
		base::vec2	uvmax;
	};

	DevTexture				tex;
	int						tile_div_x;
	int						tile_div_y;
	std::vector<TileInfo>	tiles;

	DevTileSet(const char *path,int divx,int divy,const char *tnames="")
		: tex(path), tile_div_x(divx), tile_div_y(divy), tile_names(tnames) {}

	void InitTileNames();
	void SetTileNames(const char *tn);

	void Draw(DevCanvas &c,int layer,const base::vec2 &pos,float tsize,const char *text);
	void Draw(DevCanvas &c,int layer,const base::vec2 &pos,float tsize,const int *tp,int w,int h);
    void Draw(DevCanvas &c,int layer,const base::vec2 &pos,float tsize,const int *tp,int stride,int w,int h);

private:
	std::string				tile_names;

	virtual void OnBeforeReset() {}
	virtual void OnAfterReset() {}

	virtual void OnPreload()
	{
		InitTileNames();
		SetTileNames(tile_names.c_str());
	}

    template<class T>
    void DrawInternal(DevCanvas &c,int layer,const base::vec2 &pos,float tsize,const T *tp,int stride,int end_value,int newline_value,int max_w,int max_h);

};


class CoordSpace {
public:
	enum { 
		T_FIT = 0,
		T_FILL = 1,
	};
    CoordSpace(int type,const base::vec2 &bmin,const base::vec2 &bmax,int align=0x11) : screen_size(0,0)
    { SetSpace(type,bmin,bmax,align); }

    void SetSpace(int type,const base::vec2 &bmin,const base::vec2 &bmax,int align=0x11);
	void SetCenter(const base::vec2 &_center,float _zoom=1.f) { center=_center; zoom=_zoom; Update(); }

	void SetView(base::vec2 _center,float vsize) { SetSpace(T_FIT,base::vec2(_center.x,_center.y-vsize),base::vec2(_center.x,_center.y+vsize)); }


	inline base::vec2 V2S(base::vec2 pos) { return map_world2screen.map_apply(pos); }
    inline base::vec2 S2V(base::vec2 pos) { return map_screen2world.map_apply(pos); }

	void Update();
	void OverrideScreenSize(const base::vec2 &s) { screen_size=s; }

private:
	friend class Layer;

	int			fit_mode;
	base::vec2	view_size;
	base::vec2	view_align;
	base::vec2	center;
	base::vec2	screen_size;
	float		zoom;

	base::vec4	map_world2screen;
	base::vec4	map_screen2world;
	base::vec4	map_world2ogl;

    CoordSpace();
    CoordSpace(const CoordSpace &cs);

};

struct CanvasLayerDesc {
	DevEffect	*fx;
	const char	*shader;			// technique name or microshader
	int			rstate;
	int			sampler_0;
    void		(*fn_before)();
    void		(*fn_after)();
};


class DevCanvas : public IDevResource {
public:

	// ******** Initialization ********
	//
	// - <void>												default states
	// - const char *path									path to textures
	// - const CanvasLayerDesc *ld                          list of layers
	// - const CanvasLayerDesc *ld, const char *path		list of layers & texture path

	DevCanvas() { Init(); }
	DevCanvas(const char *_tex_path) { Init(); auto_prefix=_tex_path; auto_prefix.push_back('/'); }
	DevCanvas(const CanvasLayerDesc *ld) { Init(); SetLayers(0,ld); }
	DevCanvas(const CanvasLayerDesc *ld,const char *_tex_path) {
		Init();
		auto_prefix=_tex_path;
		auto_prefix.push_back('/');
		SetLayers(0,ld);
	}
	~DevCanvas() { ClearAuto(); ClearMicroShaders(); }

	
	// Add extra layers (or replace layers) to canvas
    void SetLayers(int first_id,const CanvasLayerDesc *descs);                // terminated with shader=NULL

	// Set virtual screen (center + vertical size)
	void SetView(base::vec2 center,float vsize);

	// Coordinate conversions (virtual->screen, screen->virtual)
	base::vec2 V2S(base::vec2 pos)
	{ 
		pos -= v_center; 
		pos.x *= screen_size.x/v_size.x;
		pos.y *= screen_size.y/v_size.y;
		pos += screen_center;
		return pos;
	}

	base::vec2 S2V(base::vec2 pos)
	{
		pos -= screen_center;
		pos.x *= v_size.x/screen_size.x;
		pos.y *= v_size.y/screen_size.y;
		pos += v_center;
		return pos;
	}


	// Draw command syntax:
	//		Draw (what) (uv region) (color) (draw) ;
	//		Draw (what) (uv region) (color) .line[rep](draw) ;
	//
	// what:
	//		- <void>								- uses most recent layer and texture
	//		- int layer, DevTexture &tex			- layer & texture provided
	//		- int layer, const char *tex			- layer & auto texture
	//		- int layer, DevTileSet &tset, int tid	- layer & texture from tile set
	//		- DevTexture &tex						- layer 0 & texture provided
	//		- const char *tex						- layer 0 & auto texture
	//		- DevTileSet &tset, int tid				- layer 0 & texture from tile set
	//
	// uv region:
	//		- <void>					- texture stretched to fill
	//		- int vhflip				- texture stretched to fill + flip flags
	//		- vec2 tmin, vec2 tmax		- texture region
	//
	// color:
	//		- <void>				- white, 100% alpha
	//		- DWORD color			- RGBA color
	//
	// draw:
	//		- vec2 pos, float size, float angle		- position, size (draws square)
	//		- vec2 pos, vec2 size, float angle		- position/size
	//
	// draw .line(*):
    //		- vec2 p1, vec2 p2, float s1, float s2	                - end points, start/end width
    //		- vec2 p1, vec2 p2, float size                          - end points, width
    //		- [rep] vec2 p1, vec2 p2, float size                    - end points, width, repeats texture across length (square)
    //		- [rep] vec2 p1, vec2 p2, float size, float taspect     - end points, width, repeats texture across length (texture aspect ratio given)
    //

	struct _shape_op {
		DevCanvas &c;
		_shape_op(DevCanvas &_c) : c(_c) {}

		void operator ()(const base::vec2 &pos, float size, float angle=0)				{ c.PushQuad(pos,base::vec2(size,size),angle); }
		void operator ()(const base::vec2 &pos, const base::vec2 &size, float angle=0)	{ c.PushQuad(pos,size,angle); };

        void line(const base::vec2 &p0,const base::vec2 &p1,float s0,float s1)          { c.PushSegment(p0,p1,s0,s1); }
        void line(const base::vec2 &p0,const base::vec2 &p1,float size)                 { c.PushSegment(p0,p1,size,size); }
        void linerep(const base::vec2 &p0,const base::vec2 &p1,float size) { 
            c.active_uvmax.x = (c.active_uvmax.x-c.active_uvmin.x)*(size*2+(p1-p0).length())*(.5f/size);
            c.PushSegment(p0,p1,size,size);
        }
        void linerep(const base::vec2 &p0,const base::vec2 &p1,float size,float taspect) { 
            c.active_uvmax.x = (c.active_uvmax.x-c.active_uvmin.x)*(size*2+(p1-p0).length())*(.5f/size/taspect);
            c.PushSegment(p0,p1,size,size);
        }
    };

	struct _param_op {
		DevCanvas &c;
		_param_op(DevCanvas &_c) : c(_c) {}

		_shape_op operator ()()				{ c.active_color=0xFFFFFFFF;	return _shape_op(c); }
		_shape_op operator ()(DWORD col)	{ c.active_color=col;			return _shape_op(c); }
	};

	struct _region_op {
		DevCanvas &c;
		_region_op(DevCanvas &_c) : c(_c) {}

		_param_op operator ()()													{							return _param_op(c); }
		_param_op operator ()(int vhflip)										{ c.FlipFlags(vhflip);		return _param_op(c); }
		_param_op operator ()(const base::vec2 &tmin,const base::vec2 &tmax)	{ c.UVRegion(tmin,tmax);	return _param_op(c); }
	};

	_region_op Draw()										{ /* no op */							return _region_op(*this); }
	_region_op Draw(int layer, DevTexture &tex)				{ SelectActive(layer,tex);				return _region_op(*this); }
	_region_op Draw(int layer, const char *tex)				{ SelectActive(layer,*GetAuto(tex));	return _region_op(*this); }
	_region_op Draw(int layer, DevTileSet &tset, int tid)	{ SelectActive(layer,tset,tid);			return _region_op(*this); }
	_region_op Draw(DevTexture &tex)						{ SelectActive(0,tex);					return _region_op(*this); }
	_region_op Draw(const char *tex)						{ SelectActive(0,*GetAuto(tex));		return _region_op(*this); }
	_region_op Draw(DevTileSet &tset, int tid)				{ SelectActive(0,tset,tid);				return _region_op(*this); }



	void SelectActive(int layer,DevTexture &tex)			{ SelectLayer(layer,tex); InitUV(); }
	void SelectActive(int layer,const char *tex)			{ SelectLayer(layer,*GetAuto(tex)); InitUV(); }
	void SelectActive(int layer,DevTileSet &tset,int tile)	{ SelectLayer(layer,tset.tex); TileUV(tset,tile); }

	void Flush();

	int PushGradientQuad(const base::vec2 &pmin,const base::vec2 &pmax,const base::vec2 &uvmin,const base::vec2 &uvmax,DWORD c0,DWORD c1,DWORD c2,DWORD c3,int over=-1)
    {
        if(!active_layer) return -1;

        int vp = (over>=0) ? over : int(active_layer->size());
		if(over<0)
			active_layer->resize(vp+4);
        Vertex *v = &(*active_layer)[vp];

        v->pos = pmin;
        v->z = 0;
        v->color = c0;
        v->tc = uvmin;
        v++;

        v->pos.x = pmax.x;
        v->pos.y = pmin.y;
        v->z = 0;
        v->color = c1;
        v->tc.x = uvmax.x;
        v->tc.y = uvmin.y;
        v++;

        v->pos = pmax;
        v->z = 0;
        v->color = c3;
        v->tc = uvmax;
        v++;

        v->pos.x = pmin.x;
        v->pos.y = pmax.y;
        v->z = 0;
        v->color = c2;
        v->tc.x = uvmin.x;
        v->tc.y = uvmax.y;

		return vp;
    }


	// raw buffer access
	struct Vertex {
		enum { FVF = D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 };

		base::vec2	pos;
		float		z;
		DWORD		color;
		base::vec2	tc;
	};

	std::vector<Vertex> *GetActiveBuffer()	{ return active_layer; }


	// ******** Below this point read at your own risk ********


private:
	friend struct _region_op;
	friend struct _param_op;
	friend struct _shape_op;


	struct BatchKey {
		int			layer;
		DevTexture	*tex;

		bool operator <(const BatchKey &k) const
		{ return (layer!=k.layer) ? (layer < k.layer) : (tex < k.tex); }
	};

	struct LayerInfo {
		DevEffect	            *fx;
		std::string	            tech;
        int                     rstate;
        int                     sampler_0;
        void		            (*fn_before)();
        void		            (*fn_after)();
        std::vector<DWORD>      microshader_bin;
        IDirect3DPixelShader9   *microshader;

		LayerInfo() : fx(NULL), rstate(0), sampler_0(0), fn_before(0), fn_after(0), microshader(0) {}
	};


	typedef std::map<BatchKey,std::vector<Vertex> > tVBS;

	tVBS					vbs;
	base::vec2				screen_center;
	base::vec2				screen_size;
	base::vec2				v_center;
	base::vec2				v_size;
	std::vector<LayerInfo>	layers;
	std::vector<Vertex>		*active_layer;
	DWORD					active_color;
	base::vec2				active_uvmin;
	base::vec2				active_uvmax;
    IDirect3DVertexShader9  *micro_vs;

	std::string							auto_prefix;
	std::map<std::string,DevTexture*>	auto_tex;

	void Init();
	void SelectLayer(int layer,DevTexture &tex);

	void InitUV()
	{
		active_uvmin.x = 0;
		active_uvmin.y = 0;
		active_uvmax.x = 1;
		active_uvmax.y = 1;
	}

	void TileUV(DevTileSet &tset,int tile)
	{
		if(tile>=0 && tile<int(tset.tiles.size()))
		{
			active_uvmin = tset.tiles[tile].uvmin;
			active_uvmax = tset.tiles[tile].uvmax;
		}
	}

	void SetColor(DWORD c) { active_color = c; }

	void FlipFlags(int vhflip)
	{
		if(vhflip&1) std::swap(active_uvmin.x,active_uvmax.x);
		if(vhflip&2) std::swap(active_uvmin.y,active_uvmax.y);
	}

	void UVRegion(const base::vec2 &uvmin,const base::vec2 &uvmax)
	{
		active_uvmin = active_uvmin.get_scaled_xy(uvmax-uvmin) + uvmin;
		active_uvmax = active_uvmax.get_scaled_xy(uvmax-uvmin) + uvmin;
	}

	void PushQuad(const base::vec2 &pos,const base::vec2 &size,float angle)
	{
		assert(active_layer!=NULL);
		if(!active_layer) return;

		size_t vp = active_layer->size();
		active_layer->resize(vp+4);
		Vertex *v = &(*active_layer)[vp];
		
		float a = angle*2*D3DX_PI;
		base::vec2 du(cos(a),sin(a));
		base::vec2 dv = -du.get_rotated90();
		du *= size.x;
		dv *= size.y;


		v->pos = pos-du-dv;
		v->z = .5f;
		v->color = active_color;
		v->tc.x = active_uvmin.x;
		v->tc.y = active_uvmin.y;
		v++;

		v->pos = pos+du-dv;
		v->z = .5f;
		v->color = active_color;
		v->tc.x = active_uvmax.x;
		v->tc.y = active_uvmin.y;
		v++;

		v->pos = pos+du+dv;
		v->z = .5f;
		v->color = active_color;
		v->tc.x = active_uvmax.x;
		v->tc.y = active_uvmax.y;
		v++;

		v->pos = pos-du+dv;
		v->z = .5f;
		v->color = active_color;
		v->tc.x = active_uvmin.x;
		v->tc.y = active_uvmax.y;
	}
	
    void PushSegment(const base::vec2 &p0,const base::vec2 &p1,float s0,float s1)
    {
        assert(active_layer!=NULL);
        if(!active_layer) return;

        size_t vp = active_layer->size();
        active_layer->resize(vp+4);
        Vertex *v = &(*active_layer)[vp];

	    base::vec2 dir = p1 - p0;
	    dir.normalize();
	    base::vec2 s0_dir = dir * s0;
	    base::vec2 s1_dir = dir * s1;
	    base::vec2 s0_orthogonal = base::vec2(s0_dir.y, -s0_dir.x);
        base::vec2 s1_orthogonal = base::vec2(s1_dir.y, -s1_dir.x);


        v->pos = p0 - s0_dir + s0_orthogonal;
        v->z = .5f;
        v->color = active_color;
        v->tc.x = active_uvmin.x;
        v->tc.y = active_uvmin.y;
        v++;

        v->pos = p1 + s1_dir + s1_orthogonal;
        v->z = .5f;
        v->color = active_color;
        v->tc.x = active_uvmax.x;
        v->tc.y = active_uvmin.y;
        v++;

        v->pos = p1 + s1_dir - s1_orthogonal;
        v->z = .5f;
        v->color = active_color;
        v->tc.x = active_uvmax.x;
        v->tc.y = active_uvmax.y;
        v++;

        v->pos = p0 - s0_dir - s0_orthogonal;
        v->z = .5f;
        v->color = active_color;
        v->tc.x = active_uvmin.x;
        v->tc.y = active_uvmax.y;
    }
    
    void ClearMicroShaders();
    void BuildMicroShaders();

    void ClearAuto();
	DevTexture *GetAuto(const char *name);

    virtual void OnBeforeReset()    { ClearMicroShaders(); }
    virtual void OnAfterReset()     { BuildMicroShaders(); }
    virtual void OnPreload()        { BuildMicroShaders(); }

};


class DevTxFont : public IDevResource {
public:

	DevTxFont(const char *face);

    int  ComputeLength(const char *s,const char *e=0);
    int  GetHeight() { return height; }
	void DrawText(DevCanvas &canvas,int layer,float xp,float yp,int align,const base::vec2 &scale,int color,const char *s,const char *e=0);
	void DrawText(DevCanvas &canvas,int layer,float xp,float yp,int align,float scale,int color,const char *s,const char *e=0)
    { DrawText(canvas,layer,xp,yp,align,base::vec2(scale,scale),color,s,e); }

	void DrawTextF(DevCanvas &canvas,int layer,float xp,float yp,int align,const base::vec2 &scale,int color,const char *fmt,...);
	void DrawTextF(DevCanvas &canvas,int layer,float xp,float yp,int align,float scale,int color,const char *fmt,...);

	float DrawTextWrap(DevCanvas &canvas,int layer,float xp,float yp,float width,const base::vec2 &scale,int color,const char *s,bool nodraw=false);
	float DrawTextWrap(DevCanvas &canvas,int layer,float xp,float yp,float width,float scale,int color,const char *s,bool nodraw=false)
    { return DrawTextWrap(canvas,layer,xp,yp,width,base::vec2(scale,scale),color,s,nodraw); }

private:
	struct CharInfo {
		int		tx, ty, tw, th;
		int		ox, oy, dx;
	};

	struct KerningInfo {
		int		id, adj;

		bool operator <(const KerningInfo &k) const
		{ return id<k.id; }
	};

	std::string					path;
	std::vector<CharInfo>		chars;
	std::vector<KerningInfo>	ker;
	DevTexture					texture;
	base::vec2					tex_size;
    int							height;

	void Clear();
	bool Load(const char *name);

	DevTxFont();
	DevTxFont(const DevTxFont &);

	virtual void OnPreload() { std::string p=path; Load(p.c_str()); }
	virtual void OnBeforeReset() {}
	virtual void OnAfterReset() {}
};






class AllTweening : public base::Tracked<AllTweening>
{
};


template<class T>
class Tweening : public AllTweening
{
public:

    Tweening(const Tweening<T> &t) { *this = t.value; }
    Tweening(const T &v) { *this = v; }

    void operator =(const Tweening<T> &t) { *this = t.value; }
    void operator =(const T &v);
    void operator +=(const T &v) { *this = value + v; }
    void operator -=(const T &v) { *this = value - v; }
    void operator *=(const T &v) { *this = value * v; }
    void operator /=(const T &v) { *this = value / v; }

private:
    T       value;
    T       start_value;
    T       end_value;
    float   time;
    float   end_time;
};

// <--- back to device.cpp

using namespace std;
using namespace base;


Device Dev;



static const float CUBE_AXIS[6][3][3] = {
	{ { 0, 0,-1}, { 0, 1, 0}, { 1, 0, 0} },
	{ { 0, 0, 1}, { 0, 1, 0}, {-1, 0, 0} },
	{ { 1, 0, 0}, { 0, 0,-1}, { 0, 1, 0} },
	{ { 1, 0, 0}, { 0, 0, 1}, { 0,-1, 0} },
	{ { 1, 0, 0}, { 0, 1, 0}, { 0, 0, 1} },
	{ {-1, 0, 0}, { 0, 1, 0}, { 0, 0,-1} },
};



// **************** IDevResource ****************

IDevResource::IDevResource()
{
	DeviceResourceManager::GetInstance()->Register(this);
}

IDevResource::~IDevResource()
{
	DeviceResourceManager *drm = DeviceResourceManager::PeekInstance();
	if(drm)
		drm->Unregister(this);
}


// **************** DeviceResourceManager ****************

DeviceResourceManager				*DeviceResourceManager::singleton;
DeviceResourceManager::DRMHandler	DeviceResourceManager::handler;



void DeviceResourceManager::Register(IDevResource *res)
{
	Unregister(res);
	reslist.push_back(res);
}

void DeviceResourceManager::Unregister(IDevResource *res)
{
	for(int i=0;i<(int)reslist.size();i++)
		if(reslist[i]==res)
			reslist[i] = NULL;
	CompactTable();
}

void DeviceResourceManager::SendOnBeforeReset()
{
	for(int pr=1;pr<=16;pr++)
		for(int i=0;i<(int)reslist.size();i++)
			if(reslist[i] && reslist[i]->GetPriority()==(pr&0xF))
				reslist[i]->OnBeforeReset();
}

void DeviceResourceManager::SendOnAfterReset()
{
	for(int pr=1;pr<=16;pr++)
		for(int i=0;i<(int)reslist.size();i++)
			if(reslist[i] && reslist[i]->GetPriority()==(pr&0xF))
				reslist[i]->OnAfterReset();
}

void DeviceResourceManager::SendOnPreload()
{
	for(int pr=1;pr<=16;pr++)
		for(int i=0;i<(int)reslist.size();i++)
			if(reslist[i] && reslist[i]->GetPriority()==(pr&0xF))
				reslist[i]->OnPreload();
}

void DeviceResourceManager::SendOnReloadTick()
{
	for(int i=0;i<(int)reslist.size();i++)
		if(reslist[i])
			reslist[i]->OnReloadTick();
}

void DeviceResourceManager::CompactTable()
{
	int s=0,d=0;
	while(s<(int)reslist.size())
	{
		if(reslist[s]!=NULL)
			reslist[d++] = reslist[s];
		s++;
	}
	reslist.resize(d);
}



// **************** Device ****************


static LRESULT WINAPI _DeviceMsgProc( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam )
{
	LONG_PTR res = GetWindowLongPtr(hWnd,GWL_USERDATA);
	if(res)
		return ((Device*)res)->MsgProc(hWnd,msg,wParam,lParam);
	return DefWindowProc( hWnd, msg, wParam, lParam );
}




Device::Device() : default_font("Verdana",12,0,false,false)
{
	d3d					= NULL;
	dev					= NULL;
	d3d_hWnd			= NULL;
	d3d_need_reset		= false;
	d3d_sizing			= false;
	d3d_screen_w		= 1024;
	d3d_screen_h		=  768;
	d3d_msaa			= 0;
	d3d_windowed		= false;
	d3d_min_zstencil_w	= 0;
	d3d_min_zstencil_h	= 0;
	has_focus			= true;
	custom_zstencil		= NULL;
	custom_zstencil_rt	= NULL;
	block_surface[0]	= NULL;
	block_surface[1]	= NULL;
	block_id			= 0;

	set_resolution_first = true;
	quited = false;

	mouse_x = 0;
	mouse_y = 0;
	mouse_dx = 0;
	mouse_dy = 0;
	mouse_dz = 0;
	mouse_capture = false;

	time_prev = timeGetTime();
	time_delta = 0;
	time_elapsed = 0;
	time_first_frame = true;
	time_reload = 0;
	time_last_reload = 0;

	{
		char buff[256];
		buff[0] = 0;
		GetModuleFileName(NULL,buff,255);
		d3d_app_name = FilePathGetPart(buff,false,true,false);
		if(d3d_app_name.size()==0)
			d3d_app_name = "DirectX Application";
		else if(d3d_app_name[0]>='a' && d3d_app_name[0]<='z')
			d3d_app_name[0] += 'A' - 'a';
	}

	debug_cam_pos		= vec3(0,0,0);
	debug_cam_ypr		= vec3(0,0,0);
}

Device::~Device()
{
	Shutdown();
}

bool Device::BeginScene()
{
	has_focus = true;
	
	if(!dev)
		return false;

	HRESULT res = dev->TestCooperativeLevel();
	if(d3d_need_reset)
	{
		res = D3DERR_DEVICENOTRESET;
		d3d_need_reset = false;
	}
	if(FAILED(res))
	{
		if(res==D3DERR_DEVICENOTRESET)
		{
			SendOnBeforeReset();
			ReleaseInternalResources();

			if(FAILED(dev->Reset(GetPresentParameters())))
				return false;

			CreateInternalResources();
			SendOnAfterReset();

			dev->SetDepthStencilSurface(custom_zstencil);
			dev->SetRenderState(D3DRS_LIGHTING,FALSE);
		}
		else
			return false;
	}

	if(FAILED(dev->BeginScene()))
		return false;

	if(time_first_frame)
	{
		time_first_frame = false;
		time_prev = timeGetTime();
	}

	dprintf_buff.clear();

	if(d3d_hWnd)
		has_focus = (GetActiveWindow()==d3d_hWnd);

	return true;
}

void Device::EndScene()
{
	// debug print
	if(dprintf_buff.size()>0)
	{
		SetRenderTarget(0,NULL);
		SetDefaultRenderStates();
	
		for(int j=0;j<(int)dprintf_buff.size();j++)
		{
			for(int i=4;i>=0;i--)
			{
				static const int SEQ[6] = { 0, 0, -1, 0, 1, 0 };
				int dx = SEQ[i];
				int dy = SEQ[i+1] + j*14;
				RECT rect = { dx, dy, dx+4096, dy+4096 };
				default_font->DrawText(NULL,dprintf_buff[j].c_str(),-1,&rect,DT_TOP | DT_LEFT,i ? 0xFF000000 : 0xFFFFFFFF);
			}
		}
	}

	// end scene
	dev->EndScene();

	if(block_surface[0] && block_surface[1])
	{
		D3DLOCKED_RECT lr;
		volatile int dummy;

		static DWORD fill = 0x1234567;
		fill += 0x1243AB;
		dev->ColorFill(block_surface[block_id],0,fill);
		block_id = !block_id;

		if(!FAILED((block_surface[block_id]->LockRect(&lr,0,D3DLOCK_READONLY))))
		{
			dummy = *(int*)lr.pBits;
			block_surface[block_id]->UnlockRect();
		}
	}

	dev->Present( NULL, NULL, NULL, NULL );

	DWORD time = timeGetTime();
	time_delta = float(time-time_prev)*0.001f;
	time_prev = time;

	if(time_delta>0.3f) time_delta = 0.3f;
	time_elapsed += time_delta;

	if(time_reload>0 && time_elapsed-time_last_reload>=time_reload)
	{
		time_last_reload = time_elapsed;
		SendOnReloadTick();
	}

    // keyboard
    read_keys.clear();
}

bool Device::Sync()
{
	if(!block_surface[0])
		return false;

	D3DLOCKED_RECT lr;
	volatile int dummy;

	static DWORD fill = 0x1654123;
	fill += 0x1243AB;
	
	dev->ColorFill(block_surface[0],0,fill);

	if(FAILED((block_surface[block_id]->LockRect(&lr,0,D3DLOCK_READONLY))))
		return false;

	dummy = *(int*)lr.pBits;
	block_surface[0]->UnlockRect();

	return true;
}

void Device::SetResolution(int width,int height,bool fullscreen,int msaa)
{
	d3d_windowed = !fullscreen;

	if(!d3d_hWnd)
	{
		d3d_screen_w = width;
		d3d_screen_h = height;
		d3d_msaa = msaa;
		return;
	}

	DWORD exstyle = d3d_windowed ?	(WS_EX_APPWINDOW | WS_EX_WINDOWEDGE) :
									(WS_EX_APPWINDOW | WS_EX_TOPMOST);
	DWORD style = d3d_windowed ? WS_OVERLAPPEDWINDOW : WS_POPUP;

	SetWindowLong( d3d_hWnd, GWL_EXSTYLE, exstyle );
	SetWindowLong( d3d_hWnd, GWL_STYLE, style );

	//if(!fullscreen)
	//	ChangeDisplaySettings(NULL,0);

	RECT pos;
	if(fullscreen)
	{
		pos.left = 0;
		pos.top = 0;
	}
	else
	{
		SystemParametersInfo(SPI_GETWORKAREA,0,&pos,0);
		pos.left = (pos.left + pos.right - width)/2;
		pos.top = (pos.top + pos.bottom - height)/2;
	}
	pos.right = pos.left + width;
	pos.bottom = pos.top + height;

	AdjustWindowRectEx(&pos,style,false,exstyle);

	//if(!fullscreen)
	//	ChangeDisplaySettings(NULL,0);

	SetWindowPos( d3d_hWnd, HWND_TOP, pos.left, pos.top,
					pos.right - pos.left, pos.bottom - pos.top,SWP_SHOWWINDOW);

	//if(!fullscreen)
	//	ChangeDisplaySettings(NULL,0);

	ShowWindow( d3d_hWnd, SW_SHOW );
	UpdateWindow( d3d_hWnd );

	d3d_need_reset = true;

	if(!fullscreen && set_resolution_first)
	{
		PumpMessages();
		set_resolution_first = false;
		SetResolution(width,height,fullscreen);
		set_resolution_first = true;
	}

	set_resolution_first = true;

	POINT pt = { width/2, height/2 };
	ClientToScreen(d3d_hWnd,&pt);
	SetCursorPos(pt.x,pt.y);
}


void Device::SetAppName(const char *name)
{
	d3d_app_name = name;
	if(d3d_hWnd)
		SetWindowText(d3d_hWnd,d3d_app_name.c_str());
}


void Device::SetMinZStencilSize(int width,int height)
{
	int _w = max(d3d_screen_w,d3d_min_zstencil_w);
	int _h = max(d3d_screen_h,d3d_min_zstencil_h);
	if(width>_w || height>_h)
		d3d_need_reset = true;
	d3d_min_zstencil_w = max(width,d3d_min_zstencil_w);
	d3d_min_zstencil_h = max(height,d3d_min_zstencil_h);
}

bool Device::PumpMessages()
{
	MSG msg;

	if(quited)
		return false;

	while( PeekMessage(&msg,NULL,0,0,PM_REMOVE) )
	{
		if(msg.message==WM_QUIT)
		{
			quited = true;
			return false;
		}

		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return true;
}

bool Device::Init(bool soft_vp)
{
	const char *app_name = d3d_app_name.c_str();
	const char *class_name = "DXFW Window Class";

	// Init app name
	d3d_app_name = app_name;

	// Register window class
	WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, _DeviceMsgProc, 0L, 0L, 
						GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
						class_name, NULL };
	d3d_wc = wc;
	RegisterClassEx( &d3d_wc );

    // Create window
	int x0=0, y0=0, w=0, h=0;
    d3d_hWnd = CreateWindowEx(	d3d_windowed ?	(WS_EX_APPWINDOW | WS_EX_WINDOWEDGE) :
												(WS_EX_APPWINDOW /*| WS_EX_TOPMOST*/),	// TODO: check this
								class_name, app_name,
								d3d_windowed ? WS_OVERLAPPEDWINDOW : WS_POPUP,
								0, 0, d3d_screen_w, d3d_screen_h,
								NULL, NULL, d3d_wc.hInstance, NULL );

	SetWindowLongPtr(d3d_hWnd,GWL_USERDATA,(LONG_PTR)this);
	has_focus = true;

	// position and init window
	SetResolution(d3d_screen_w,d3d_screen_h,!d3d_windowed,d3d_msaa);

	// Create Direct3D
	if( NULL == ( d3d = Direct3DCreate9( D3D_SDK_VERSION ) ) )
	{
		MessageBox(NULL,"Can't initialize Direct3D!","Error!",MB_OK);
		Shutdown();
		return false;
	}

	// Create device
	DWORD flags = 0;
	if(soft_vp)	flags |= D3DCREATE_SOFTWARE_VERTEXPROCESSING;
	else		flags |= D3DCREATE_HARDWARE_VERTEXPROCESSING;
	if( FAILED( d3d->CreateDevice( D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, d3d_hWnd,
						flags, GetPresentParameters(), &dev ) ) )
	{
		MessageBox(NULL,"Can't initialize Direct3D!","Error!",MB_OK);
		Shutdown();
		return false;
	}

	// Set device state
	dev->SetRenderState(D3DRS_LIGHTING,FALSE);

	ShowWindow( d3d_hWnd, SW_SHOW );
	UpdateWindow( d3d_hWnd );

	// Create z/stencil & locking surfaces
	CreateInternalResources();

	PreloadResources();

    // Init raw input
    RAWINPUTDEVICE device;
    device.usUsagePage = 0x01;
    device.usUsage = 0x02;
    device.dwFlags = 0;
    device.hwndTarget = 0;
    RegisterRawInputDevices(&device, 1, sizeof(device));

	return true;
}

void Device::Shutdown()
{
	ReleaseInternalResources();

	if(dev) dev->Release();		dev = NULL;
	if(d3d) d3d->Release();		d3d = NULL;
	DestroyWindow(d3d_hWnd);	d3d_hWnd = NULL;

	UnregisterClass( "DXFW Window Class", d3d_wc.hInstance );
}



bool Device::MainLoop()
{
	if(!d3d_hWnd)
	{
		if(!Dev.Init())
			return false;
	}
	else
		Dev.EndScene();

	while(1)
	{
		if(!Dev.PumpMessages())
		{
			Dev.Shutdown();
			return false;
		}

		if(Dev.BeginScene())
			break;

		Sleep(300);
	}

	return true;
}



void Device::SetDefaultRenderStates()
{
	// vertex processing
	dev->SetRenderState(D3DRS_LIGHTING,FALSE);
	dev->SetVertexShader(NULL);

	// z/stencil
	dev->SetRenderState(D3DRS_ZENABLE,TRUE);
	dev->SetRenderState(D3DRS_ZFUNC,D3DCMP_LESSEQUAL);
	dev->SetRenderState(D3DRS_ZWRITEENABLE,TRUE);
	dev->SetRenderState(D3DRS_SCISSORTESTENABLE,FALSE);
	dev->SetRenderState(D3DRS_STENCILENABLE,FALSE);

	// pixel processing
	dev->SetPixelShader(NULL);

	dev->SetTextureStageState(0,D3DTSS_COLOROP,D3DTOP_MODULATE);
	dev->SetTextureStageState(0,D3DTSS_COLORARG1,D3DTA_TEXTURE);
	dev->SetTextureStageState(0,D3DTSS_COLORARG2,D3DTA_DIFFUSE);
	dev->SetTextureStageState(0,D3DTSS_ALPHAOP,D3DTOP_MODULATE);
	dev->SetTextureStageState(0,D3DTSS_ALPHAARG1,D3DTA_TEXTURE);
	dev->SetTextureStageState(0,D3DTSS_ALPHAARG2,D3DTA_DIFFUSE);
	dev->SetTextureStageState(1,D3DTSS_COLOROP,D3DTOP_DISABLE);
	
	dev->SetTexture(0,NULL);

	// framebuffer
	dev->SetRenderState(D3DRS_ALPHABLENDENABLE,FALSE);
	dev->SetRenderState(D3DRS_BLENDOP,D3DBLENDOP_ADD);

	dev->SetRenderState(D3DRS_ALPHATESTENABLE,FALSE);

	dev->SetRenderState(D3DRS_COLORWRITEENABLE,0xF);
}

void Device::SetRState(int mode)
{
    Dev->SetRenderState(D3DRS_ZENABLE,(mode & RSF_NO_ZENABLE)==0);
    Dev->SetRenderState(D3DRS_ZWRITEENABLE,(mode & RSF_NO_ZWRITE)==0);
    Dev->SetRenderState(D3DRS_CULLMODE,((mode>>2)&3)+1);
    Dev->SetRenderState(D3DRS_ALPHATESTENABLE,(mode&0x00FF0000)!=0);
    if(mode&0x00FF0000)
    {
        Dev->SetRenderState(D3DRS_ALPHAFUNC,D3DCMP_GREATEREQUAL);
        Dev->SetRenderState(D3DRS_ALPHAREF,(mode>>16)&0xFF);
    }
    Dev->SetRenderState(D3DRS_ALPHABLENDENABLE,(mode&0xFF000070)!=0);
    if(mode&0xFF000070)
    {
        Dev->SetRenderState(D3DRS_SRCBLEND,((mode>>24)&0xF)+1);
        Dev->SetRenderState(D3DRS_DESTBLEND,((mode>>28)&0xF)+1);
        Dev->SetRenderState(D3DRS_BLENDOP,(mode>>4)&7);
    }
    Dev->SetRenderState(D3DRS_COLORWRITEENABLE,(~(mode>>8))&15);
}

void Device::SetSampler(int id,int flags)
{
	static const int minmag[5] = { D3DTEXF_POINT,	D3DTEXF_LINEAR,	D3DTEXF_LINEAR, D3DTEXF_LINEAR,	D3DTEXF_ANISOTROPIC };
	static const int mip[5]    = { D3DTEXF_NONE,	D3DTEXF_NONE,	D3DTEXF_POINT,	D3DTEXF_LINEAR,	D3DTEXF_LINEAR };
    int texf = flags & 0x1F;
    int texa = (flags >> 5)&0x7;
    if(!texa) texa = D3DTADDRESS_WRAP;

	int t = (texf<=4) ? texf : 4;
	dev->SetSamplerState(id,D3DSAMP_MAGFILTER,minmag[t]);
	dev->SetSamplerState(id,D3DSAMP_MINFILTER,minmag[t]);
	dev->SetSamplerState(id,D3DSAMP_MIPFILTER,mip[t]);
	if(t>=4) dev->SetSamplerState(id,D3DSAMP_MAXANISOTROPY,t-2);
	dev->SetSamplerState(id,D3DSAMP_ADDRESSU,texa);
	dev->SetSamplerState(id,D3DSAMP_ADDRESSV,texa);
	dev->SetSamplerState(id,D3DSAMP_ADDRESSW,texa);
    if(texa==TEXA_BORDER) dev->SetSamplerState(id,D3DSAMP_BORDERCOLOR,(flags&TEXB_WHITE) ? 0xFFFFFFFF : 0x00000000);
}

void Device::SetRenderTarget(int id,DevRenderTarget *rt)
{
	if(id==0 && custom_zstencil_rt)
		dev->SetDepthStencilSurface( rt ? custom_zstencil_rt : custom_zstencil );

	if(!rt)
	{
		if(id==0)
		{
			IDirect3DSurface9 *surf = NULL;
			dev->GetBackBuffer(0,0,D3DBACKBUFFER_TYPE_MONO,&surf);
			if(surf)
			{
				dev->SetRenderTarget(0,surf);
				surf->Release();
			}
		}
		else
			dev->SetRenderTarget(id,NULL);
		return;
	}

	IDirect3DSurface9 *surf = rt->GetSurface();
	if(!surf) return;
	dev->SetRenderTarget(id,surf);
	surf->Release();
}


void Device::SetCubeRenderTarget(int id,DevCubeRenderTarget *rt,int face)
{
	if(id==0 && custom_zstencil_rt)
		dev->SetDepthStencilSurface( rt ? custom_zstencil_rt : custom_zstencil );

	if(!rt)
	{
		SetRenderTarget(id,NULL);
		return;
	}

	IDirect3DSurface9 *surf = rt->GetSurface(face);
	if(!surf) return;
	dev->SetRenderTarget(id,surf);
	surf->Release();
}


void Device::SetDepthStencil(DevDepthStencilSurface *ds)
{
	if(!ds)
	{
		dev->SetDepthStencilSurface(custom_zstencil);
		return;
	}

	dev->SetDepthStencilSurface(ds->GetSurface());
}

void Device::StretchRect(DevRenderTarget *src,const RECT *srect,
						 DevRenderTarget *dst,const RECT *drect,
						 D3DTEXTUREFILTERTYPE filter)
{
	IDirect3DSurface9 *ssurf = NULL;
	IDirect3DSurface9 *dsurf = NULL;
	
	if(src)	ssurf = src->GetSurface();
	else	dev->GetBackBuffer(0,0,D3DBACKBUFFER_TYPE_MONO,&ssurf);

	if(dst)	dsurf = dst->GetSurface();
	else	dev->GetBackBuffer(0,0,D3DBACKBUFFER_TYPE_MONO,&dsurf);

	if(ssurf && dsurf)
	{
		dev->StretchRect(ssurf,srect,dsurf,drect,filter);
	}

	if(ssurf) ssurf->Release();
	if(dsurf) dsurf->Release();
}

void Device::DrawScreenQuad(float x1,float y1,float x2,float y2,float z)
{
	struct Vertex {
		float x, y, z, w;
		float u, v;
	};

	Vertex vtx[4];
	for(int i=0;i<4;i++)
	{
		vtx[i].x = ((i==0 || i==3) ? x1 : x2) - 0.5f;
		vtx[i].y = ((i==0 || i==1) ? y1 : y2) - 0.5f;
		vtx[i].z = z;
		vtx[i].w = 1;
		vtx[i].u = ((i==0 || i==3) ? 0.f : 1.f);
		vtx[i].v = ((i==0 || i==1) ? 0.f : 1.f);
	}

	dev->SetFVF(D3DFVF_XYZRHW|D3DFVF_TEX1);
	dev->DrawPrimitiveUP(D3DPT_TRIANGLEFAN,2,vtx,sizeof(Vertex));
}

void Device::DrawScreenQuadTC(float x1,float y1,float x2,float y2,float *coord_rects,int n_coords,float z)
{
	struct Vertex {
		float x, y, z, w;
		vec2 uv[8];
	};

	Vertex vtx[4];
	for(int i=0;i<4;i++)
	{
		vtx[i].x = ((i==0 || i==3) ? x1 : x2) - 0.5f;
		vtx[i].y = ((i==0 || i==1) ? y1 : y2) - 0.5f;
		vtx[i].z = z;
		vtx[i].w = 1;
		for(int j=0;j<n_coords;j++)
		{
			vtx[i].uv[j].x = ((i==0 || i==3) ? coord_rects[j*4+0] : coord_rects[j*4+2]);
			vtx[i].uv[j].y = ((i==0 || i==1) ? coord_rects[j*4+1] : coord_rects[j*4+3]);
		}

	}

	dev->SetFVF(D3DFVF_XYZRHW|(D3DFVF_TEX1*n_coords));
	dev->DrawPrimitiveUP(D3DPT_TRIANGLEFAN,2,vtx,sizeof(Vertex));
}

void Device::DrawScreenQuadTCI(float x1,float y1,float x2,float y2,float *tc00,float *tc10,float *tc01,float *tc11,
								int n_interp,int tc_fvf,float z)
{
	struct Vertex {
		float x, y, z, w;
		float uv[8*4];
	};

	Vertex vtx[4];
	float dd = 0;

	if(!(tc_fvf & D3DFVF_POSITION_MASK))
	{
		tc_fvf |= D3DFVF_XYZRHW;
		dd = 0.5f;
	}

	for(int i=0;i<4;i++)
	{
		vtx[i].x = ((i==0 || i==3) ? x1 : x2) - dd;
		vtx[i].y = ((i==0 || i==1) ? y1 : y2) - dd;
		vtx[i].z = z;
		vtx[i].w = 1;
	}
	memcpy(vtx[0].uv,tc00,sizeof(float)*n_interp);
	memcpy(vtx[1].uv,tc10,sizeof(float)*n_interp);
	memcpy(vtx[2].uv,tc11,sizeof(float)*n_interp);
	memcpy(vtx[3].uv,tc01,sizeof(float)*n_interp);

	dev->SetFVF(tc_fvf);
	dev->DrawPrimitiveUP(D3DPT_TRIANGLEFAN,2,vtx,sizeof(Vertex));
}

void Device::DrawScreenQuadTCIR(float x1,float y1,float x2,float y2,float *rects,int n_interp,int tc_fvf,float z)
{
	struct Vertex {
		float x, y, z, w;
		float uv[8*4];
	};

	Vertex vtx[4];
	float dd = 0;

	if(!(tc_fvf & D3DFVF_POSITION_MASK))
	{
		tc_fvf |= D3DFVF_XYZRHW;
		dd = 0.5f;
	}

	for(int i=0;i<4;i++)
	{
		vtx[i].x = ((i==0 || i==3) ? x1 : x2) - dd;
		vtx[i].y = ((i==0 || i==1) ? y1 : y2) - dd;
		vtx[i].z = z;
		vtx[i].w = 1;
		for(int j=0;j<n_interp;j++)
			vtx[i].uv[j] = rects[j*4+(i^(i>>1))];
	}

	dev->SetFVF(tc_fvf);
	dev->DrawPrimitiveUP(D3DPT_TRIANGLEFAN,2,vtx,sizeof(Vertex));
}

void Device::DrawScreenQuadVS(float x1,float y1,float x2,float y2,float z)
{
	struct Vertex {
		float x, y, z;
		float u, v;
	};

	Vertex vtx[4];
	for(int i=0;i<4;i++)
	{
		vtx[i].x = ((i==0 || i==3) ? x1 : x2);
		vtx[i].y = ((i==0 || i==1) ? y1 : y2);
		vtx[i].z = z;
		vtx[i].u = ((i==0 || i==3) ? 0.f : 1.f);
		vtx[i].v = ((i==0 || i==1) ? 0.f : 1.f);
	}

	dev->SetFVF(D3DFVF_XYZ|D3DFVF_TEX1);
	dev->DrawPrimitiveUP(D3DPT_TRIANGLEFAN,2,vtx,sizeof(Vertex));
}

void Device::DrawSkyBoxQuad(const vec3 &ypr,float fov)
{
	vec3 vc, vx, vy;
	Dev.BuildCameraVectors(ypr,&vc,&vx,&vy);
	vc *= 1.f/tanf(fov*(0.5f*D3DX_PI/180.f));
	vy = -vy;
	vx *= Dev.GetAspectRatio();

	int sw, sh;
	Dev.GetScreenSize(sw,sh);

	vec3 TC[4] = {
		vc-vx-vy, vc+vx-vy,
		vc-vx+vy, vc+vx+vy,
	};

	Dev.DrawScreenQuadTCI(0,0,float(sw),float(sh),&TC[0].x,&TC[1].x,&TC[2].x,&TC[3].x,3,
							D3DFVF_TEX1|D3DFVF_TEXCOORDSIZE3(0),1.f - 1.f/(1<<16));
}


void Device::Print(DevFont *fnt,int xp,int yp,int color,const char *text)
{
	RECT rect = { xp, yp, xp+4096, yp+4096 };
	if(!fnt) fnt = &default_font;
	fnt->GetFont()->DrawText(NULL,text,-1,&rect,DT_TOP | DT_LEFT,color);
}

void Device::PrintF(DevFont *fnt,int xp,int yp,int color,const char *fmt, ...)
{
	va_list arg;
	string tmp;
	va_start(arg,fmt);
	vsprintf(tmp,fmt,arg);
	va_end(arg);

	Print(fnt,xp,yp,color,tmp.c_str());
}

void Device::AlignPrint(DevFont *fnt,int xp,int yp,int align,int color,const char *text)
{
	int ax = (align>>4)&3;
	int ay = align&3;
	DWORD fl = DT_SINGLELINE;
	RECT rect = { xp, yp, xp, yp };
	if(ax>=1) rect.left		-= 2048;
	if(ax<=1) rect.right	+= 2048;
	if(ay>=1) rect.top		-= 2048;
	if(ay<=1) rect.bottom	+= 2048;
	if(ax==0) fl |= DT_LEFT;
	if(ax==1) fl |= DT_CENTER;
	if(ax==2) fl |= DT_RIGHT;
	if(ay==0) fl |= DT_TOP;
	if(ay==1) fl |= DT_VCENTER;
	if(ay==2) fl |= DT_BOTTOM;

	if(!fnt) fnt = &default_font;
	fnt->GetFont()->DrawText(NULL,text,-1,&rect,fl,color);
}

void Device::AlignPrintF(DevFont *fnt,int xp,int yp,int color,int align,const char *fmt, ...)
{
	va_list arg;
	string tmp;
	va_start(arg,fmt);
	vsprintf(tmp,fmt,arg);
	va_end(arg);

	AlignPrint(fnt,xp,yp,color,align,tmp.c_str());
}


void Device::DPrintF(const char *fmt, ...)
{
	va_list arg;
	string tmp;
	va_start(arg,fmt);
	vsprintf(tmp,fmt,arg);
	va_end(arg);

	dprintf_buff.push_back(tmp);
}

int *Device::GetQuadsIndices(int n_quads)
{
	if((int)quad_idx.size()<n_quads*6)
	{
		static int IDX[6] = {0,1,3,3,1,2};
		int p = quad_idx.size();
		quad_idx.resize(n_quads*6);
		while(p<(int)quad_idx.size())
		{
			quad_idx[p] = (p/6)*4 + IDX[p%6];
			p++;
		}
	}
	return quad_idx.size() ? &quad_idx[0] : NULL;
}


void Device::SetMouseCapture(bool capture)
{
    if(capture && !mouse_capture)
    {
		SetCursor(NULL);
		POINT pt = { d3d_screen_w/2, d3d_screen_h/2 };
		ClientToScreen(d3d_hWnd,&pt);
		SetCursorPos(pt.x,pt.y);
    }

    mouse_capture = capture;
}

bool Device::GetKeyStroke(int vk)
{
    for(int i=0;i<(int)read_keys.size();i++)
        if(read_keys[i]==vk)
        {
            read_keys.erase(read_keys.begin()+i);
            return true;
        }

    return false;
}



void Device::RegisterMessageHandler(IDevMsgHandler *dmh)
{
	UnregisterMessageHandler(dmh);
	msg_handlers.push_back(dmh);
}

void Device::UnregisterMessageHandler(IDevMsgHandler *dmh)
{
	for(int i=0;i<(int)msg_handlers.size();i++)
		if(msg_handlers[i]==dmh)
		{
			msg_handlers.erase(msg_handlers.begin()+i);
			i--;
		}
}



// ------------


D3DPRESENT_PARAMETERS *Device::GetPresentParameters()
{
	ZeroMemory( &d3dpp, sizeof(d3dpp) );
	d3dpp.Windowed					= d3d_windowed;
	d3dpp.BackBufferWidth			= d3d_screen_w;
	d3dpp.BackBufferHeight			= d3d_screen_h;
	d3dpp.BackBufferFormat			= D3DFMT_A8R8G8B8;
	d3dpp.BackBufferCount			= 1;
	d3dpp.EnableAutoDepthStencil	= FALSE;
	d3dpp.AutoDepthStencilFormat	= D3DFMT_UNKNOWN;
	d3dpp.SwapEffect				= D3DSWAPEFFECT_DISCARD;
	d3dpp.Flags						= 0;
	d3dpp.MultiSampleType			= (D3DMULTISAMPLE_TYPE)d3d_msaa;
	d3dpp.MultiSampleQuality		= 0;
	d3dpp.hDeviceWindow				= d3d_hWnd;
	d3dpp.FullScreen_RefreshRateInHz= 0;
	d3dpp.PresentationInterval		= D3DPRESENT_INTERVAL_ONE;	// D3DPRESENT_INTERVAL_DEFAULT;
	return &d3dpp;
}

LRESULT Device::MsgProc( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam )
{
	for(int i=0;i<(int)msg_handlers.size();i++)
	{
		bool pass = true;
		LRESULT res = msg_handlers[i]->MsgProc(this,hWnd,msg,wParam,lParam,pass);
		if(!pass) return res;
	}

	switch( msg )
	{
		case WM_DESTROY:
			PostQuitMessage( 0 );
			return 0;

		case WM_SIZE:
			d3d_screen_w = LOWORD(lParam);
			d3d_screen_h = HIWORD(lParam);
			d3d_need_reset = true;
		return 0;

		case WM_SIZING:
			{
				RECT *box = (RECT*)lParam;
				RECT wr,cr;
				GetWindowRect(hWnd,&wr);
				GetClientRect(hWnd,&cr);

				// compute frames width
				wr.left -= cr.left;
				wr.right -= cr.right;
				wr.top -= cr.top;
				wr.bottom -= cr.bottom;

				// compute new client rect
				cr = *box;
				cr.left -= wr.left;
				cr.right -= wr.right;
				cr.top -= wr.top;
				cr.bottom -= wr.bottom;

				// compute client rect
				cr.right -= cr.left;
				cr.bottom -= cr.top;
				cr.right = (cr.right+16)&~0x1F;
				cr.bottom = (cr.bottom+16)&~0x1F;
				if(cr.right<64) cr.right = 64;
				if(cr.bottom<64) cr.bottom = 64;
				cr.right += cr.left;
				cr.bottom += cr.top;

				// compute new window rect
				cr.left += wr.left;
				cr.right += wr.right;
				cr.top += wr.top;
				cr.bottom += wr.bottom;
				*box = cr;

				d3d_sizing = true;
				UpdateWindow(hWnd);
			}
		return 1;

		case WM_PAINT:
			if(d3d_sizing)
			{
				HDC hDC = GetDC(hWnd);
				RECT rect;
				rect.top = rect.left = 0;
				rect.bottom = rect.right = 4096;
				FillRect(hDC,&rect,(HBRUSH)GetStockObject(BLACK_BRUSH));
				ReleaseDC(hWnd,hDC);
				d3d_sizing = false;
				return 0;
			}
		break;

		case WM_MOUSEMOVE:
			{
				if(!mouse_capture)
				{
					mouse_x = LOWORD(lParam);
					mouse_y = HIWORD(lParam);
					SetCursor(LoadCursor(NULL,IDC_ARROW));
				}
				else
				{
					int mx = LOWORD(lParam);
					int my = HIWORD(lParam);

					SetCursor(NULL);
					int dx = mx - d3d_screen_w/2;
					int dy = my - d3d_screen_h/2;
					mouse_dx += dx;
					mouse_dy += dy;
					if(dx || dy)
					{
						POINT pt = { d3d_screen_w/2, d3d_screen_h/2 };
						ClientToScreen(d3d_hWnd,&pt);
						SetCursorPos(pt.x,pt.y);
					}
				}
			}
		return 0;

		case WM_MOUSEWHEEL:
			{
				mouse_dz += short(HIWORD(wParam))/float(WHEEL_DELTA);
			}
		return 0;

		case WM_LBUTTONDOWN:	read_keys.push_back(VK_LBUTTON);	return 0;
		case WM_RBUTTONDOWN:	read_keys.push_back(VK_RBUTTON);	return 0;
		case WM_MBUTTONDOWN:	read_keys.push_back(VK_MBUTTON);	return 0;

		case WM_SYSKEYDOWN:
			if(wParam==VK_F4 && ((GetAsyncKeyState(VK_LMENU)<0) || (GetAsyncKeyState(VK_RMENU)<0)))
				DestroyWindow(hWnd);
			read_keys.push_back(wParam);
		return 0;

		case WM_KEYDOWN:
			read_keys.push_back(wParam);
/*			if(wParam=='1') D3D_SetResolution( 640, 480, true);
			if(wParam=='2') D3D_SetResolution( 800, 600, true);
			if(wParam=='3') D3D_SetResolution(1024, 768, true);
			if(wParam=='4') D3D_SetResolution(1280,1024, true);
			if(wParam=='Q') D3D_SetResolution( 640, 480,false);
			if(wParam=='W') D3D_SetResolution( 800, 600,false);
			if(wParam=='E') D3D_SetResolution(1024, 768,false);
			if(wParam=='R') D3D_SetResolution(1280,1024,false);
*/
		return 0;

		case WM_CHAR:
			read_keys.push_back(~wParam);
		return 0;

		case WM_SYSCHAR:
			read_keys.push_back(wParam | 0x10000);
		return TRUE;

        case WM_INPUT:
        {
            static vector<byte> rawInputMessageData;

            bool inForeground = (GET_RAWINPUT_CODE_WPARAM(wParam) == RIM_INPUT);
            HRAWINPUT hRawInput = (HRAWINPUT)lParam;

            UINT dataSize;
            UINT u = GetRawInputData(hRawInput, RID_INPUT, NULL, &dataSize, sizeof(RAWINPUTHEADER));

            if( u==0 && dataSize != 0)
            {
                rawInputMessageData.resize(dataSize);
                
                void* dataBuf = &rawInputMessageData[0];

                if( GetRawInputData(hRawInput, RID_INPUT, dataBuf, &dataSize, sizeof(RAWINPUTHEADER)) == dataSize )
                {
                    const RAWINPUT *raw = (const RAWINPUT*)dataBuf;

                    if (raw->header.dwType == RIM_TYPEMOUSE)
                    {
                        const RAWMOUSE& mouseData = raw->data.mouse;
                        HANDLE deviceHandle = raw->header.hDevice;
                        USHORT flags = mouseData.usButtonFlags;
                        short wheelDelta = (short)mouseData.usButtonData;
                        LONG x = mouseData.lLastX, y = mouseData.lLastY;

                        size_t index = 0;
                        while(index < mouse_data.size() && mouse_data[index].handle != deviceHandle)
                            ++index;
                        if(index >= mouse_data.size())
                        {
                            mouse_data.push_back(RawMouse());
                            mouse_data[index].Clear();
                            mouse_data[index].handle = deviceHandle;
                        }

                        RawMouse &data = mouse_data[index];
                        data.xp += x;
                        data.yp += y;
                        data.dx += x;
                        data.dy += y;
                        data.dz += wheelDelta;

                        double tt = Dev.GetElapsedTime();
                        for(int b=0;b<5;b++)
                        {
                            bool press   = (flags & (1<<(b*2)))!=0;
                            bool release = (flags & (2<<(b*2)))!=0;
                            if(press)
                            {
                                data.bt_down  |= 1<<b;
                                data.bt_click |= 1<<b;
                                if(tt - data.press_time[b] <= data.dbclick_timeout)
                                    data.bt_2click |= 1<<b;
                                data.press_time[b] = tt;
                            }
                            if(release)
                            {
                                data.bt_down &= ~(1<<b);
                            }
                        }
                    }
                }
            }
        }
        break;
	}

	return DefWindowProc( hWnd, msg, wParam, lParam );
}

void Device::CreateInternalResources()
{
	ReleaseInternalResources();

	for(int i=0;i<2;i++)
		dev->CreateOffscreenPlainSurface(16,16,D3DFMT_A8R8G8B8,D3DPOOL_DEFAULT,&block_surface[i],0);

	dev->CreateDepthStencilSurface(
		max(d3d_screen_w,d3d_min_zstencil_w),max(d3d_screen_h,d3d_min_zstencil_h),
		D3DFMT_D24S8, (D3DMULTISAMPLE_TYPE)d3d_msaa, 0, FALSE, &custom_zstencil, NULL );

	if(d3d_msaa)
	{
		dev->CreateDepthStencilSurface(
			max(d3d_screen_w,d3d_min_zstencil_w),max(d3d_screen_h,d3d_min_zstencil_h),
			D3DFMT_D24S8, D3DMULTISAMPLE_NONE, 0, FALSE, &custom_zstencil_rt, NULL );
	}

	dev->SetDepthStencilSurface(custom_zstencil);

//	if(!custom_zstencil)
//		MessageBox(NULL,format("Failed to create primary z/stencil surface (code %08x)",res).c_str(),"Error!",MB_OK);
}

void Device::ReleaseInternalResources()
{
	for(int i=0;i<2;i++)
		if(block_surface[i])
		{
			block_surface[i]->Release();
			block_surface[i] = NULL;
		}

	if(custom_zstencil)
	{
		dev->SetDepthStencilSurface(NULL);
		custom_zstencil->Release();
		custom_zstencil = NULL;
	}

	if(custom_zstencil_rt)
	{
		dev->SetDepthStencilSurface(NULL);
		custom_zstencil_rt->Release();
		custom_zstencil_rt = NULL;
	}

}



// -------- helper functions --------


void Device::BuildProjectionMatrix(D3DXMATRIX *out,float vport[4],int rt_size_x,int rt_size_y,
								   const vec3 &ax,const vec3 &ay,const vec3 &az,float zn,float zf)
{
	static float DEFAULT_VPORT[4] = { -1, -1, 1, 1 };

	if(!vport)
		vport = DEFAULT_VPORT;

	float _vp[4] = { vport[0], vport[1], vport[2], vport[3] };
	float s = 1.0f;
	_vp[0] -= s/rt_size_x;
	_vp[1] += s/rt_size_y;
	_vp[2] -= s/rt_size_x;
	_vp[3] += s/rt_size_y;

	float sx = 2/(_vp[2] - _vp[0]);
	float sy = 2/(_vp[3] - _vp[1]);
	float dx = -(_vp[2] + _vp[0])/(_vp[2] - _vp[0]);
	float dy = -(_vp[3] + _vp[1])/(_vp[3] - _vp[1]);

	out->_11 = sx*ax.x - dx*az.x;
	out->_21 = sx*ax.y - dx*az.y;
	out->_31 = sx*ax.z - dx*az.z;
	out->_41 = 0.f;

	out->_12 = sy*ay.x - dy*az.x;
	out->_22 = sy*ay.y - dy*az.y;
	out->_32 = sy*ay.z - dy*az.z;
	out->_42 = 0.f;

	float Q = zf/(zf-zn);
	if(zf<=0)
		Q = (zf<0) ? (1 - 1.f/(1<<16)) : (1 + 1.f/(1<<16));
	out->_13 = Q*az.x;
	out->_23 = Q*az.y;
	out->_33 = Q*az.z;
	out->_43 = -Q*zn;

	out->_14 = az.x;
	out->_24 = az.y;
	out->_34 = az.z;
	out->_44 = 0.f;
}

void Device::BuildCubemapProjectionMatrix(D3DXMATRIX *mtx,int face,int size,float zn,float zf,float *vport)
{
	BuildProjectionMatrix(mtx,vport,size,size,
		*(vec3*)CUBE_AXIS[face][0],
		*(vec3*)CUBE_AXIS[face][1],
		*(vec3*)CUBE_AXIS[face][2],
		zn,zf);
}

void Device::BuildCameraViewProjMatrix(D3DXMATRIX *mtx,const vec3 &pos,const vec3 &_ypr,float fov,
									float aspect,float zn,float zf,bool m_view,bool m_proj)
{
	if(!m_view && !m_proj)
	{
		D3DXMatrixIdentity(mtx);
		return;
	}

	D3DXMATRIX view, proj;
	D3DXMATRIX *p = &proj;

	if(m_view)
	{
		D3DXVECTOR3 up(0,0,1), target;

		vec3 ypr = _ypr*(D3DX_PI/180);
		target.x = cosf(ypr.x)*cosf(ypr.y);
		target.y = sinf(ypr.x)*cosf(ypr.y);
		target.z = sinf(ypr.y);
		target += *(D3DXVECTOR3*)&pos;

		if(!m_proj)
		{
			D3DXMatrixLookAtLH(mtx,(D3DXVECTOR3*)&pos,(D3DXVECTOR3*)&target,&up);
			return;
		}

		D3DXMatrixLookAtLH(&view,(D3DXVECTOR3*)&pos,(D3DXVECTOR3*)&target,&up);
	}
	else
		p = mtx;

	if(aspect<=0)
		aspect = GetAspectRatio();
	D3DXMatrixPerspectiveFovLH(p,fov*(D3DX_PI/180.f),aspect,zn,zf);
	
	if(zf<=0)
	{
		float Q = (zf<0) ? (1 - 1.f/(1<<16)) : (1 + 1.f/(1<<16));
		p->_33 = Q;
		p->_43 = -Q*zn;
	}

	if(m_view && m_proj)
		D3DXMatrixMultiply(mtx,&view,&proj);
}

void Device::BuildCameraVectors(const vec3 &ypr,vec3 *front,vec3 *right,vec3 *up)
{
	float ax = ypr.x*(D3DX_PI/180);
	float ay = ypr.y*(D3DX_PI/180);
	if(front)
	{
		front->x = cosf(ax)*cosf(ay);
		front->y = sinf(ax)*cosf(ay);
		front->z = sinf(ay);
	}
	if(right)
	{
		right->x = -sinf(ax);
		right->y =  cosf(ax);
		right->z =  0;
	}
	if(up)
	{
		up->x = -cosf(ax)*sinf(ay);
		up->y = -sinf(ax)*sinf(ay);
		up->z =  cosf(ay);
	}
}


void Device::TickFPSCamera(vec3 &pos,vec3 &ypr,float move_speed,float sens,bool flat)
{
	vec3 d(0,0,0);
	move_speed *= GetTimeDelta();
	if(GetKeyState('W')) d.x += move_speed;
	if(GetKeyState('S')) d.x -= move_speed;
	if(GetKeyState('A')) d.y -= move_speed;
	if(GetKeyState('D')) d.y += move_speed;
	if(GetKeyState('Q')) d.z -= move_speed;
	if(GetKeyState('E')) d.z += move_speed;

	float ax = ypr.x*(D3DX_PI/180);
	if(flat)
	{
		pos.x += cosf(ax)*d.x - sinf(ax)*d.y;
		pos.y += cosf(ax)*d.y + sinf(ax)*d.x;
		pos.z += d.z;
	}
	else
	{
		vec3 front, right;
		BuildCameraVectors(ypr,&front,&right,NULL);
		pos += front*d.x + right*d.y;
		pos.z += d.z;
	}

	int mx, my;
	SetMouseCapture(true);
	GetMouseDelta(mx,my);
	ypr.x += mx*sens;
	ypr.y -= my*sens;

	if(ypr.y<-89.9f) ypr.y= -89.9f;
	if(ypr.y> 89.9f) ypr.y=  89.9f;

}

void Device::RunDebugCamera(float speed,float sens,float zn,float fov,bool flat)
{
	TickFPSCamera(debug_cam_pos,debug_cam_ypr,speed,sens,flat);

	D3DXMATRIX vp;
	
	BuildCameraViewProjMatrix(&vp,debug_cam_pos,debug_cam_ypr,fov,0,zn,-1,true,false);
	dev->SetTransform(D3DTS_VIEW,&vp);

	BuildCameraViewProjMatrix(&vp,debug_cam_pos,debug_cam_ypr,fov,0,zn,-1,false,true);
	dev->SetTransform(D3DTS_PROJECTION,&vp);
}

// ******************************** d_canvas.cpp ********************************

// ---- #include "dxfw.h"
// ---> including dxfw.h
// <--- back to d_canvas.cpp

using namespace std;
using namespace base;


//
// Microshader data:
//  s[0-3]      sampler     - texture samplers
//  c           float4      - input color param
//  t           float4      - color from texture
//  o           float4      - output color (default: t*c)
//  uv          float2      - texcoord
//  wpos        float2      - virtual (world) position
//  vpos        float2      - screen position in pixels
//  spos        float2      - normalized screen position (for full screen rendertarget sampling)
//
// Microshader functions:
//  mask        - true if given color is close to color (value is 0..1 or 1..255)
//  killmask    - as above, but kills pixel if true
//


static const char *MICRO_PREFIX =
"sampler s0 : register(s0);\n"
"sampler s1 : register(s1);\n"
"sampler s2 : register(s2);\n"
"sampler s3 : register(s3);\n"
"float4 v2s : register(c30);\n"
"float2 ssize : register(c31);\n"
"float fix(float x) { return x<=1 ? x : x/255; }\n"
"bool mask(float4 x,float r,float g,float b,float e) { return dot(abs(x.xyz-float3(fix(r),fix(g),fix(b))),1)<fix(e); }\n"
"void killmask(float4 x,float r,float g,float b,float e) { if(mask(x,r,g,b,e)) discard; }\n"
"float4 PS(float4 c : COLOR0, float2 uv : TEXCOORD0, float2 wpos : TEXCOORD1) : COLOR0\n"
"{ float4 t=tex2D(s0,uv), o=t*c;\n"
"  float2 vpos = wpos*v2s.wz+v2s.xy;\n"
"  float2 spos = vpos/ssize;\n";

static const char *MICRO_SUFFIX =
"\nreturn o;\n"
"}\n";


static const char *MICRO_VS =
"float4 v2s : register(c30);\n"
"float2 ssize : register(c31);\n"
"void VS( float4 pos : POSITION, float4 col : COLOR, float2 uv : TEXCOORD0,\n"
"         out float4 hpos : POSITION, out float4 _col : COLOR0, out float2 _uv : TEXCOORD0, out float2 _wpos : TEXCOORD1)\n"
"{hpos.xy=(pos.xy*v2s.wz+v2s.xy)/ssize*2-1;\n"
" hpos.y*=-1;\n"
" hpos.zw=float2(.5,1);\n"
" _col=col; _uv=uv; _wpos=pos.xy;\n"
"}\n";




// ---------------- DevTileSet ----------------


void DevTileSet::InitTileNames()
{
	tiles.clear();

	for(int y=0;y<tile_div_y;y++)
		for(int x=0;x<tile_div_x;x++)
		{
			TileInfo ti;
			ti.uvmin.x = float(x  )/tile_div_x;
			ti.uvmin.y = float(y  )/tile_div_y;
			ti.uvmax.x = float(x+1)/tile_div_x;
			ti.uvmax.y = float(y+1)/tile_div_y;
			tiles.push_back(ti);
		}
}

void DevTileSet::SetTileNames(const char *tn)
{
	int tx = 0, ty = 0;
	while(*tn)
	{
		int c = int(*tn)&0xFF;

		if(*tn=='\n') tx=0, ty++;
		if(*tn=='\r') tx=0, ty=0;

		if(c>=' ')
		{
			if(c>=(int)tiles.size())
				tiles.resize(c+1);
			
			TileInfo &ti = tiles[c];
			ti.uvmin.x = float(tx  )/tile_div_x;
			ti.uvmin.y = float(ty  )/tile_div_y;
			ti.uvmax.x = float(tx+1)/tile_div_x;
			ti.uvmax.y = float(ty+1)/tile_div_y;
			tx++;
		}

		tn++;
	}
}

template<class T>
void DevTileSet::DrawInternal(DevCanvas &c,int layer,const base::vec2 &pos,float tsize,const T *tp,int stride,int end_value,int newline_value,int max_w,int max_h)
{
    int xp=0, yp=0;

    while(1)
    {
        if(xp==max_w) xp=0, yp++;
        if(yp==max_h) break;

        int v = *tp;
        if(v==end_value) break;
        
        if(v==newline_value)
        {
            xp=0, yp++;
            if(yp==max_h) break;
        }
        else
        {
            if(v>=0 && v<(int)tiles.size())
                c.Draw(layer,*this,v)()()(pos+vec2(xp+.5f,yp+.5f)*tsize,tsize*.5f);
            xp++;
        }

        *(byte**)&tp += stride;
    }
}

void DevTileSet::Draw(DevCanvas &c,int layer,const base::vec2 &pos,float tsize,const char *text)
{
    DrawInternal(c,layer,pos,tsize,text,sizeof(char),0,'\n',-1,-1);
}

void DevTileSet::Draw(DevCanvas &c,int layer,const base::vec2 &pos,float tsize,const int *tp,int w,int h)
{
    DrawInternal(c,layer,pos,tsize,tp,sizeof(int),0x80000000,0x80000000,w,h);
}

void DevTileSet::Draw(DevCanvas &c,int layer,const base::vec2 &pos,float tsize,const int *tp,int stride,int w,int h)
{
    DrawInternal(c,layer,pos,tsize,tp,stride,0x80000000,0x80000000,w,h);
}



// ---------------- CoordSpace ----------------


void CoordSpace::SetSpace(int type,const vec2 &bmin,const vec2 &bmax,int align)
{
	fit_mode = type;
	view_size = bmax - bmin;
	center = (bmin+bmax)*.5f;
	zoom = 1;
	view_align.x =  align/16 /2.f;
	view_align.y = (align%16)/2.f;

	Update();
}

void CoordSpace::Update()
{
	vec4 w2u, u2v, v2s, s2hw;
	vec2 ss = screen_size.x>0 ? screen_size : Dev.GetScreenSizeV();

	float z = .5f/zoom;
	w2u.make_map_to_unit( center - view_size*z, center + view_size*z );
	u2v.make_map_unit_to( vec2(0,0), view_size );
	if(fit_mode==T_FIT)
	{
		vec4 tmp;
		tmp.make_map_box_scale_fit( ss, view_size, view_align );
		v2s.make_map_inverse(tmp);
	}
	else
		v2s.make_map_box_scale_fit( view_size, ss, view_align );
	s2hw.make_map_to_view( vec2(0,0), ss );

	map_world2screen = w2u;
	map_world2screen.make_map_concat(map_world2screen,u2v);
	map_world2screen.make_map_concat(map_world2screen,v2s);
	map_world2ogl = map_world2screen;
	
	map_screen2world.make_map_inverse(map_world2screen);
}


// ---------------- DevCanvas ----------------

void DevCanvas::Init()
{
	active_layer = NULL;
	active_color = 0xFFFFFFFF;
    micro_vs = NULL;
}


void DevCanvas::SetLayers(int first_id,const CanvasLayerDesc *descs)
{
    for(int i=0;descs[i].shader;i++)
    {
        int id = first_id + i;

        if(id>=int(layers.size()))
            layers.resize(id+1);

        const CanvasLayerDesc &d = descs[i];
        LayerInfo &li = layers[id];

        li.fx = d.fx;
        li.tech = d.shader;
        li.rstate = d.rstate;
        li.sampler_0 = d.sampler_0;
        li.fn_before = d.fn_before;
        li.fn_after = d.fn_after;

		if(li.microshader)
		{
			li.microshader->Release();
			li.microshader = NULL;
		}
		li.microshader_bin.clear();
	}

    BuildMicroShaders();
}


void DevCanvas::SetView(base::vec2 center,float vsize)
{
	screen_size = Dev.GetScreenSizeV();
	screen_center = screen_size*0.5f - vec2(.5f,.5f);
	v_center = center;
	v_size.y = vsize;
	v_size.x = vsize/screen_size.y*screen_size.x;
}

void DevCanvas::SelectLayer(int layer,DevTexture &tex)
{
	BatchKey key;
	key.layer = layer;
	key.tex = &tex;

    if(layer>=int(layers.size()))
        layers.resize(layer+1);
    
    active_layer = &vbs[key];
}

void DevCanvas::Flush()
{
	tVBS::iterator p;
	Dev->SetFVF(Vertex::FVF);
	Dev->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_MODULATE);
	Dev->SetTextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
	Dev->SetTextureStageState(0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE);
	Dev->SetTextureStageState(0, D3DTSS_COLOROP, D3DTOP_MODULATE);
	Dev->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
	Dev->SetTextureStageState(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE);

    // shader setup
    vec2 v2s_add = V2S(vec2(0,0));
    vec2 v2s_mul = V2S(vec2(1,1)) - v2s_add;
    vec4 shdata[2] = {
        vec4(v2s_add.x,v2s_add.y,v2s_mul.y,v2s_mul.x),
        vec4(screen_size.x,screen_size.y,0,0)
    };


    Dev->SetVertexShader(micro_vs);
    Dev->SetVertexShaderConstantF(30,&shdata[0].x,2);
    Dev->SetPixelShaderConstantF(30,&shdata[0].x,2);
	
	for(p=vbs.begin();p!=vbs.end();p++)
    {
        LayerInfo &li = layers[p->first.layer];

        if(li.fn_before)
            li.fn_before();

        if(p->second.size()>0 && p->first.layer>=0 && p->first.layer<(int)layers.size())
		{
			DevTexture *tex = p->first.tex;

            Dev.SetRState(li.rstate);
            Dev.SetSampler(0,li.sampler_0);
			Dev->SetTexture(0,tex ? tex->GetTexture() : NULL);

			if(li.fx)
			{
				(*li.fx)->SetTexture("tex",tex ? tex->GetTexture() : NULL);
				li.fx->StartTechnique(li.tech.c_str());

				while(li.fx->StartPass())
				{
					Dev->SetFVF( Vertex::FVF );
					Dev->DrawIndexedPrimitiveUP(
						D3DPT_TRIANGLELIST,0,int(p->second.size()),int(p->second.size())/4*2,
						Dev.GetQuadsIndices(int(p->second.size())/4),D3DFMT_INDEX32,
						&p->second[0],sizeof(Vertex));
				}
            }
			else
			{
                Dev->SetPixelShader(li.microshader);

                Dev->SetFVF( Vertex::FVF );
				Dev->DrawIndexedPrimitiveUP(
					D3DPT_TRIANGLELIST,0,int(p->second.size()),int(p->second.size())/4*2,
					Dev.GetQuadsIndices(int(p->second.size())/4),D3DFMT_INDEX32,
					&p->second[0],sizeof(Vertex));

                Dev->SetPixelShader(NULL);
			}

			p->second.clear();
		}

        if(li.fn_after)
            li.fn_after();
    }

    Dev->SetVertexShader(NULL);
}

void DevCanvas::ClearMicroShaders()
{
    for(int i=0;i<(int)layers.size();i++)
    {
        LayerInfo &l = layers[i];

        if(l.microshader)
        {
            l.microshader->Release();
            l.microshader = NULL;
        }
    }

    if(micro_vs)
    {
        micro_vs->Release();
        micro_vs = NULL;
    }
}

void DevCanvas::BuildMicroShaders()
{
    for(int i=0;i<(int)layers.size();i++)
    {
        LayerInfo &l = layers[i];
        if(l.fx || l.tech.size()<=0) continue;

        if(l.microshader_bin.size()<=0)
        {
            // Try compiling shader bytecode.
            string code = MICRO_PREFIX;
            code += l.tech;
            code += MICRO_SUFFIX;

            ID3DXBuffer *bin=0, *err=0;
            if(FAILED(D3DXCompileShader(code.c_str(),(UINT)code.size(),0,0,"PS","ps_2_0",0,&bin,&err,0)) || !bin || err)
            {
                const char *error = (err ? (const char*)err->GetBufferPointer() : "Unknown error");
                string e = error;
                e += "\n\nwhile compiling microshader:\n";

                const char *s = code.c_str();
                int line = 1;
                while(*s)
                {
                    e += format("[%2d] ",line++);
                    const char *b = s;
                    while(*s && *s!='\n') s++;
                    if(*s=='\n') s++;
                    e.append(b,s);
                }

                if(MessageBox(0,e.c_str(),"Microshader error!",MB_OKCANCEL)==IDCANCEL)
                    ExitProcess(0);

                l.microshader_bin.resize(0);
                l.microshader_bin[0] = 0xFFFFFFFF;
            }
            else
            {
                int ndwords = bin->GetBufferSize()/4;
                l.microshader_bin.resize(ndwords);
                memcpy(&l.microshader_bin[0],bin->GetBufferPointer(),ndwords*sizeof(DWORD));
            }

            if(bin) bin->Release();
            if(err) err->Release();
        }

        if(Dev.GetIsReady() && !l.microshader && l.microshader_bin.size()>0 && l.microshader_bin[0]!=0xFFFFFFFF)
        {
            HRESULT res = Dev->CreatePixelShader(&l.microshader_bin[0],&l.microshader);

			assert( SUCCEEDED(res) );
        }
    }

    if(!micro_vs)
    {
        // Try compiling shader bytecode.

        ID3DXBuffer *bin=0, *err=0;
        if(FAILED(D3DXCompileShader(MICRO_VS,strlen(MICRO_VS),0,0,"VS","vs_2_0",0,&bin,&err,0)) || !bin || err)
        {
            const char *error = (err ? (const char*)err->GetBufferPointer() : "Unknown error");
            string e = error;
            e += "\n\nwhile compiling micro VS:\n";

            const char *s = MICRO_VS;
            int line = 1;
            while(*s)
            {
                e += format("[%2d] ",line++);
                const char *b = s;
                while(*s && *s!='\n') s++;
                if(*s=='\n') s++;
                e.append(b,s);
            }

            if(MessageBox(0,e.c_str(),"Microshader error!",MB_OKCANCEL)==IDCANCEL)
                ExitProcess(0);
        }
        else
        {
            if(Dev.GetIsReady() && !micro_vs)
            {
                HRESULT res = Dev->CreateVertexShader((DWORD*)bin->GetBufferPointer(),&micro_vs);

                assert( SUCCEEDED(res) );
            }
        }

        if(bin) bin->Release();
        if(err) err->Release();
    }
}

void DevCanvas::ClearAuto()
{
	for(map<string,DevTexture*>::iterator p=auto_tex.begin();p!=auto_tex.end();p++)
		if(p->second)
			delete p->second;
	auto_tex.clear();
}

DevTexture *DevCanvas::GetAuto(const char *name)
{
	map<string,DevTexture*>::iterator p = auto_tex.find(name);
	if(p!=auto_tex.end())
		return p->second;

	// load texture
	DevTexture *tex = new DevTexture();

	if(!tex->Load((auto_prefix+name).c_str()))
	{
		delete tex;
		tex = NULL;
	}

	auto_tex[name] = tex;
	return tex;
}

// ******************************** d_effect.cpp ********************************

// ---- #include "dxfw.h"
// ---> including dxfw.h
// <--- back to d_effect.cpp

using namespace std;
using namespace base;



// **************** DevEffect ****************


DevEffect::DevEffect(const char *path,DevEffect::DefGenerator *dg,int _flags) : fx(NULL), pool(NULL)
{
	do_preload = false;
	pass = -2;
	def_generator = dg;
	flags = _flags;
	file_time = 0;

	compile_flags = D3DXSHADER_AVOID_FLOW_CONTROL;
	if( flags & FXF_FLOW_CONTROL_MEDIUM )	compile_flags &= ~D3DXSHADER_AVOID_FLOW_CONTROL;
	if( flags & FXF_FLOW_CONTROL_HIGH	)	compile_flags |= D3DXSHADER_PREFER_FLOW_CONTROL;
	if( flags & FXF_PARTIAL_PRECISION	)	compile_flags |= D3DXSHADER_PARTIALPRECISION;
		 if( flags & FXF_OPTIMIZATION_3	)	compile_flags |= D3DXSHADER_OPTIMIZATION_LEVEL3;
	else if( flags & FXF_OPTIMIZATION_0	)	compile_flags |= D3DXSHADER_OPTIMIZATION_LEVEL0;
	else if( flags & FXF_OPTIMIZATION_1	)	compile_flags |= D3DXSHADER_OPTIMIZATION_LEVEL1;
	else if( flags & FXF_OPTIMIZATION_2	)	compile_flags |= D3DXSHADER_OPTIMIZATION_LEVEL2;
    if( flags & FXF_LEGACY              )	compile_flags |= D3DXSHADER_USE_LEGACY_D3DX9_31_DLL;



	if(Dev.GetIsReady())	Load(path,dg);
	else					do_preload = true, p_path = path;
}

bool DevEffect::Load(const char *path,DevEffect::DefGenerator *dg)
{
	// clear effect
	ClearEffect();
	file_time = GetFileTime(path);
	p_path = string(path).c_str();

	// load file
	vector<char> data;
	if(!ScrambleLoad(path,data))
		return false;

	// build skip list
	string skip_list;
	BuildSkipList(&data[0],skip_list);

	// build defines
	vector<string> defs;
	if(dg)	dg(defs);
	else	defs.push_back(":");

	// create pool
	if(defs.size()>=2)
	{
		if(FAILED(D3DXCreateEffectPool(&pool)))
		{
			return false;
		}
	}

	// compile
	for(int i=0;i<(int)defs.size();i++)
	{
		vector<D3DXMACRO> macros;
		vector<byte> bin;

		defs[i].push_back(0);

		// get macros
		ShatterDefsList(&defs[i][0],macros);

		// build binary path
		string bpath = FilePathGetPart((string("shaders/")+path).c_str(),true,true,false);
		if(defs[i].c_str()[0])
		{
			bpath += "-";
			bpath += defs[i].c_str();
		}
		bpath += ".fxx";

		// precompile effect
		byte bhash[16] = {};
		if(!GetPrecompiledBinary(data,defs[i].c_str(),path,bpath.c_str(),macros,bhash,bin))
			return false;

		if(!CompileVersion(defs[i].c_str(),path,bin,macros,skip_list.c_str()))
			return false;
	}

	if(fx_list.size()<=0)
		return false;

	fx = fx_list[0];

	// load textures
	D3DXHANDLE h, ha;
	const char *str;
	int id = 0;
	while( (h = fx->GetParameter(NULL,id)) )
	{
		ha = fx->GetAnnotationByName(h,"Load");
		if(ha)
			if(SUCCEEDED(fx->GetString(ha,&str)))
			{
				DevTexture *t = new DevTexture();
				t->Load(str);
				fx->SetTexture(h,t->GetTexture());
				textures.push_back(t);
			}
		id++;
	}

	return true;
}

const char *DevEffect::GetTechniqueName(int id)
{
	D3DXHANDLE h;
	D3DXTECHNIQUE_DESC desc;
	h = fx->GetTechnique(id);
	if(!h) return NULL;
	if(SUCCEEDED(fx->GetTechniqueDesc(h,&desc)))
		return desc.Name;
	return NULL;
}

const char *DevEffect::GetTechniqueInfo(const char *name,const char *param)
{
	D3DXHANDLE h, ha;
	const char *str;
	h = fx->GetTechniqueByName(name);
	if(!h) return "";
	ha = fx->GetAnnotationByName(h,param);
	if(!ha) return "";
	if(SUCCEEDED(fx->GetString(ha,&str)))
		return str;
	return "";
}

bool DevEffect::SelectVersionStr(const char *version)
{
	map<string,int>::iterator p = version_index.find(version);
	if(p==version_index.end()) { fx = NULL; return false; }
	fx = fx_list[p->second];
	return true;
}

bool DevEffect::SelectVersion(int version)
{
	if(version<0 || version>=(int)fx_list.size()) { fx = NULL; return false; }
	fx = fx_list[version];
	return true;
}

int DevEffect::GetVersionIndex(const char *version)
{
	map<string,int>::iterator p = version_index.find(version);
	if(p==version_index.end()) return -1;
	return p->second;
}


bool DevEffect::StartTechnique(const char *name)
{
	if(!fx)
		return false;

	if(FAILED(fx->SetTechnique(name)))
	{
		pass = -2;
		return false;
	}
	pass = -1;

	return true;
}

bool DevEffect::StartPass()
{
	if(pass==-2)
		return false;

	if(pass<0)	fx->Begin((UINT*)&n_passes,0);
	else		fx->EndPass();
	pass++;

	if(pass>=n_passes)
	{
		fx->End();
		pass = -2;
		return false;
	}

	fx->BeginPass(pass);

	return true;
}

void DevEffect::OnBeforeReset()
{
	for(int i=0;i<(int)fx_list.size();i++)
		fx_list[i]->OnLostDevice();
}

void DevEffect::OnAfterReset()
{
	for(int i=0;i<(int)fx_list.size();i++)
		fx_list[i]->OnResetDevice();
}


void DevEffect::ClearEffect()
{
	for(int i=0;i<(int)fx_list.size();i++)
		fx_list[i]->Release();
	fx_list.clear();
	fx = NULL;
	if(pool)
	{
		pool->Release();
		pool = NULL;
	}
	version_index.clear();
	do_preload = false;
	pass = -2;
	last_error.clear();
}

bool DevEffect::ScrambleLoad(const char *path,vector<char> &data)
{
	bool scrambled = false;
	if(!NFS.GetFileBytes(path,*(vector<byte>*)&data) || data.size()==0)
	{
		last_error = format("Can't find file %s!",path);
		if(!(flags&FXF_NO_ERROR_POPUPS))
			if(MessageBox(NULL,last_error.c_str(),path,MB_OKCANCEL)==IDCANCEL)
				ExitProcess(0);
		return false;
	}

	data.push_back(0);

	return true;
}

void DevEffect::BuildSkipList(const char *d,string &skip_list)
{
	while(*d)
	{
		if(d[0]=='/' && d[1]=='*' && d[2]=='$' && d[3]=='*' && d[4]=='/' && d[5]==' ')
		{
			const char *p = d+6;
			if(skip_list.size()>0) skip_list.push_back(';');
			while((*p>='a' && *p<='z') || (*p>='A' && *p<='Z') || (*p>='0' && *p<='9') || *p=='_')
				skip_list.push_back(*p++);
		}
		d++;
	}
}

void DevEffect::ShatterDefsList(char *s,vector<D3DXMACRO> &macros)
{
	// parse version identifier
	const char *id = s;
	while(*s && *s!=':') s++;

	// parse macros
	while(*s)
	{
		D3DXMACRO m = { NULL, NULL };

		// terminate previous string
		*s++ = 0;
		
		// read name
		m.Name = s;
		while(*s && *s!='=') s++;
		if(!*s) break;
		*s++ = 0;

		// read definition
		m.Definition = s;
		while(*s && *s!=';') s++;

		macros.push_back(m);
	}

	// terminate macro list
	D3DXMACRO m = { NULL, NULL };
	macros.push_back(m);
}

bool DevEffect::GetPrecompiledBinary(
		vector<char> &data, const char *id, const char *path, const char *bpath,
		vector<D3DXMACRO> &macros, byte hash[16], vector<byte> &bin)
{
	ID3DXEffectCompiler *fxc = NULL;
	ID3DXBuffer *binbuff = NULL;
	ID3DXBuffer *errors = NULL;

	for(int pass=0;pass<2;pass++)
	{
		HRESULT r;
		bool error;
		
		if( pass == 0 )
		{
			r = D3DXCreateEffectCompiler(&data[0],(DWORD)data.size(),&macros[0],NULL,
					compile_flags,&fxc,&errors);
			error = !fxc;
		}
		else
		{
			r = fxc->CompileEffect(compile_flags,&binbuff,&errors);
			error = !binbuff;
		}

		if(FAILED(r) || errors || error)
		{
			if(errors)
			{
				last_error = format("%s:%s: %s",path,id,(char*)errors->GetBufferPointer());
				if(!(flags&FXF_NO_ERROR_POPUPS))
					if(MessageBox(NULL,last_error.c_str(),format("%s:%s",path,id).c_str(),MB_OKCANCEL)==IDCANCEL)
						ExitProcess(0);
				errors->Release();
				if(fxc) fxc->Release();
				if(binbuff) binbuff->Release();
				return false;
			}
			else
			{
				last_error = format("%s:%s: Unknown error!",path,id);
				if(!(flags&FXF_NO_ERROR_POPUPS))
					if(MessageBox(NULL,last_error.c_str(),format("%s:%s",path,id).c_str(),MB_OKCANCEL)==IDCANCEL)
						ExitProcess(0);
				if(fxc) fxc->Release();
				if(binbuff) binbuff->Release();
				return false;
			}
		}
	}

	bin.clear();
	bin.insert(bin.end(),(byte*)binbuff->GetBufferPointer(),((byte*)binbuff->GetBufferPointer())+binbuff->GetBufferSize());

	fxc->Release();
	binbuff->Release();

	return true;
}

bool DevEffect::CompileVersion(
			const char *id, const char *path, vector<byte> &bin,
			vector<D3DXMACRO> &macros, const char *skip_list )
{
	ID3DXEffect *fx = NULL;
	ID3DXBuffer *errors = NULL;

	HRESULT r = D3DXCreateEffectEx(Dev.GetDevice(),&bin[0],(DWORD)bin.size(),&macros[0],NULL,skip_list,
		compile_flags,pool,&fx,&errors);

	if(FAILED(r) || errors)
	{
		if(errors)
		{
			last_error = format("%s:%s: %s",path,id,(char*)errors->GetBufferPointer());
			if(!(flags&FXF_NO_ERROR_POPUPS))
				if(MessageBox(NULL,last_error.c_str(),format("%s:%s",path,id).c_str(),MB_OKCANCEL)==IDCANCEL)
					ExitProcess(0);
			errors->Release();
			return false;
		}
		else
		{
			last_error = format("%s:%s: Unknown error!",path,id);
			if(!(flags&FXF_NO_ERROR_POPUPS))
				if(MessageBox(NULL,last_error.c_str(),format("%s:%s",path,id).c_str(),MB_OKCANCEL)==IDCANCEL)
					ExitProcess(0);
			return false;
		}
	}

	version_index[id] = (int)fx_list.size();
	fx_list.push_back(fx);

	return true;
}

// ******************************** d_formats.cpp ********************************

// ---- #include "dxfw.h"
// ---> including dxfw.h
// <--- back to d_formats.cpp

using namespace std;
using namespace base;



static void _conv_A8R8G8B8_u(DWORD rgba,void *out)				{ *(DWORD*)out = rgba; }
static void _conv_R5G6B5_u(DWORD rgba,void *out)				{ *(word*)out = ((rgba>>8)&0xF800) | ((rgba>>5)&0x07E0) | ((rgba>>3)&0x001F); }
static void _conv_A1R5G5B5_u(DWORD rgba,void *out)				{ *(word*)out = ((rgba>>31)&0x8000) | ((rgba>>9)&0x7C00) | ((rgba>>6)&0x03E0) | ((rgba>>3)&0x001F); }
static void _conv_X1R5G5B5_u(DWORD rgba,void *out)				{ *(word*)out = ((rgba>>9)&0x7C00) | ((rgba>>6)&0x03E0) | ((rgba>>3)&0x001F); }
static void _conv_A4R4G4B4_u(DWORD rgba,void *out)				{ *(word*)out = ((rgba>>16)&0xF000) | ((rgba>>12)&0x0F00) | ((rgba>>8)&0x00F0) | ((rgba>>4)&0x000F); }
static void _conv_X4R4G4B4_u(DWORD rgba,void *out)				{ *(word*)out = ((rgba>>12)&0x0F00) | ((rgba>>8)&0x00F0) | ((rgba>>4)&0x000F); }
static void _conv_A8R3G3B2_u(DWORD rgba,void *out)				{ *(word*)out = ((rgba>>16)&0xFF00) | ((rgba>>16)&0x00E0) | ((rgba>>11)&0x001C) | ((rgba>>6)&0x0003); }
static void _conv_R3G3B2_u(DWORD rgba,void *out)				{ *(byte*)out = ((rgba>>16)&0xE0) | ((rgba>>11)&0x1C) | ((rgba>>6)&0x03); }
static void _conv_A8_u(DWORD rgba,void *out)					{ *(byte*)out = rgba>>24; }
static void _conv_L8_u(DWORD rgba,void *out)					{ *(byte*)out = byte(rgba); }
static void _conv_A8L8_u(DWORD rgba,void *out)					{ *(word*)out = (rgba>>16) | (rgba&0xFF); }

static void _conv_A8R8G8B8_f(const float *rgba,void *out)		{ *(DWORD*)out = ((vec3*)rgba)->make_rgba(rgba[3]); }
static void _conv_X8R8G8B8_f(const float *rgba,void *out)		{ *(DWORD*)out = ((vec3*)rgba)->make_rgb(); }
static void _conv_R5G6B5_f(const float *rgba,void *out)			{ DWORD tmp = ((vec3*)rgba)->make_rgb(); 
																  *(word*)out = ((tmp>>8)&0xF800) | ((tmp>>5)&0x07E0) | ((tmp>>3)&0x001F);
																}
static void _conv_A8_f(const float *rgba,void *out)				{ float tmp = rgba[3]*(255.f/256.f)+1.0f; *(byte*)out = byte((*(DWORD*)&tmp)>>15); }
static void _conv_L8_f(const float *rgba,void *out)				{ float tmp = rgba[0]*(255.f/256.f)+1.0f; *(byte*)out = byte((*(DWORD*)&tmp)>>15); }
static void _conv_L16_f(const float *rgba,void *out)			{ float tmp = rgba[0]*(65535.f/65536.f)+1.0f; *(word*)out = word((*(DWORD*)&tmp)>>7); }
static void _conv_A8L8_f(const float *rgba,void *out)			{ float l = rgba[0]*(255.f/256.f)+1.0f; 
																  float a = rgba[3]*(255.f/256.f)+1.0f;
																  *(word*)out = (((*(DWORD*)&a)>>7)&0xFF00) | (((*(DWORD*)&l)>>15)&0x00FF);
																}
static void _conv_G16R16_f(const float *rgba,void *out)			{ float r = rgba[0]*(65535.f/65536.f)+1.0f;
																  float g = rgba[1]*(65535.f/65536.f)+1.0f;
																  *(DWORD*)out = ((*(DWORD*)&g)<<9); *(word*)out = word((*(DWORD*)&r)>>7);
																}
static void _conv_A16B16G16R16_f(const float *rgba,void *out)	{ float r = rgba[0]*(65535.f/65536.f)+1.0f;
																  float g = rgba[1]*(65535.f/65536.f)+1.0f;
																  float b = rgba[2]*(65535.f/65536.f)+1.0f;
																  float a = rgba[3]*(65535.f/65536.f)+1.0f;
																  *(DWORD*)out = ((*(DWORD*)&g)<<9); *(word*)out = word((*(DWORD*)&r)>>7);
																  ((DWORD*)out)[1] = ((*(DWORD*)&a)<<9); ((word*)out)[2] = word((*(DWORD*)&b)>>7);
																}
static void _conv_R32F_f(const float *rgba,void *out)			{ *(float*)out = rgba[0]; }
static void _conv_G32R32F_f(const float *rgba,void *out)		{ *(int64*)out = *(int64*)rgba; }
static void _conv_A32B32G32R32F_f(const float *rgba,void *out)	{ *(int64*)out = *(int64*)rgba; ((int64*)out)[1] = ((int64*)rgba)[1]; }




struct FormatDef {
	D3DFORMAT	format;
	char		*descr;
	int			bpp;
	void		(*fn_convert_u)(DWORD rgba,void *out);
	void		(*fn_convert_f)(const float *rgba,void *out);
};


#define FMT(name)	D3DFMT_ ## name, #name

static const FormatDef _FORMATS[] = {
	// BackBuffer, Display and Unsigned
	{ FMT( A2R10G10B10	), 4	,NULL	, NULL	},
	{ FMT( A8R8G8B8		), 4	,_conv_A8R8G8B8_u	,_conv_A8R8G8B8_f		},
	{ FMT( X8R8G8B8		), 4	,_conv_A8R8G8B8_u	,_conv_X8R8G8B8_f		},
	{ FMT( A1R5G5B5		), 2	,_conv_A1R5G5B5_u	,NULL					},
	{ FMT( X1R5G5B5		), 2	,_conv_X1R5G5B5_u	,NULL					},
	{ FMT( R5G6B5		), 2	,_conv_R5G6B5_u		,_conv_R5G6B5_f			},
//	R8G8B8
	{ FMT( A4R4G4B4		), 2	,_conv_A4R4G4B4_u	,NULL					},
	{ FMT( R3G3B2		), 1	,_conv_R3G3B2_u		,NULL					},
	{ FMT( A8			), 1	,_conv_A8_u			,_conv_A8_f				},
	{ FMT( A8R3G3B2		), 2	,_conv_A8R3G3B2_u	,NULL					},
	{ FMT( X4R4G4B4		), 2	,_conv_X4R4G4B4_u	,NULL					},
//	A2B10G10R10
//	A8B8G8R8
//	X8B8G8R8
	{ FMT( G16R16		), 4	,NULL				,_conv_G16R16_f			},
	{ FMT( A16B16G16R16	), 8	,NULL				,_conv_A16B16G16R16_f	},
//	A8P8
//	P8
	{ FMT( L8			), 1	,_conv_L8_u			,_conv_L8_f				},
	{ FMT( L16			), 2	,NULL				,_conv_L16_f			},
	{ FMT( A8L8			), 2	,_conv_A8L8_u		,_conv_A8L8_f			},
	{ FMT( A4L4			), 1	,NULL				,NULL					},

	// Depth-Stencil
	{ FMT( D16_LOCKABLE	), 2	,NULL				,NULL					},
//	D32
//	D15S1
	{ FMT( D24S8		), 0	,NULL				,NULL					},
	{ FMT( D24X8		), 0	,NULL				,NULL					},
//	D24X4S4
	{ FMT( D32F_LOCKABLE), 4	,NULL				,NULL					},
//	D24FS8
	{ FMT( D16			), 0	,NULL				,NULL					},
	// DXTn Compressed
	{ FMT( DXT1			), 0	,NULL				,NULL					},
	{ FMT( DXT2			), 0	,NULL				,NULL					},
	{ FMT( DXT3			), 0	,NULL				,NULL					},
	{ FMT( DXT4			), 0	,NULL				,NULL					},
	{ FMT( DXT5			), 0	,NULL				,NULL					},
	// Floating-Point
	{ FMT( R16F			), 2	,NULL				,NULL					},
	{ FMT( G16R16F		), 4	,NULL				,NULL					},
	{ FMT( A16B16G16R16F), 8	,NULL				,NULL					},
	// FOURCC
//	MULTI2_ARGB8
//	G8R8_G8B8
//	R8G8_B8G8
//	UYVY
//	YUY2
	// IEEE
	{ FMT( R32F			), 4	,NULL				,_conv_R32F_f			},
	{ FMT( G32R32F		), 8	,NULL				,_conv_G32R32F_f		},
	{ FMT( A32B32G32R32F),16	,NULL				,_conv_A32B32G32R32F_f	},
	// Mixed
//	L6V5U5
//	X8L8V8U8
//	A2W10V10U10
	// Signed
	{ FMT( V8U8			), 2	,NULL				,NULL					},
	{ FMT( Q8W8V8U8		), 4	,NULL				,NULL					},
	{ FMT( V16U16		), 4	,NULL				,NULL					},
	{ FMT( Q16W16V16U16	), 8	,NULL				,NULL					},
	{ FMT( CxV8U8		), 2	,NULL				,NULL					},

	// End
	{ D3DFMT_UNKNOWN, NULL, }
};

#undef FMT


static const FormatDef *_find_format(D3DFORMAT format)
{
	const FormatDef *fd = _FORMATS;
	while(fd->format!=D3DFMT_UNKNOWN)
	{
		if(fd->format==format)
			return fd;
		fd++;
	}
	return NULL;
}



int Device::GetSurfaceSize(D3DFORMAT format,int w,int h)
{
	const FormatDef *fd = _find_format(format);
	return fd ? (w*h*fd->bpp) : 0;
}

Device::fn_convert_u_t *Device::GetFormatConversionFunctionU(D3DFORMAT format)
{
	const FormatDef *fd = _find_format(format);
	return fd ? fd->fn_convert_u : NULL;
}

Device::fn_convert_f_t *Device::GetFormatConversionFunctionF(D3DFORMAT format)
{
	const FormatDef *fd = _find_format(format);
	return fd ? fd->fn_convert_f : NULL;
}

bool Device::IsFormatDepthStencil(D3DFORMAT format)
{
	return (format==D3DFMT_D24S8		|| format==D3DFMT_D16			|| format==D3DFMT_D24X8  ||
			format==D3DFMT_D16_LOCKABLE	|| format==D3DFMT_D24X4S4		|| format==D3DFMT_D15S1  ||
			format==D3DFMT_D32			|| format==D3DFMT_D32F_LOCKABLE || format==D3DFMT_D24FS8	);
}

// ******************************** d_mesh.cpp ********************************

// ---- #include "dxfw.h"
// ---> including dxfw.h
// <--- back to d_mesh.cpp

using namespace std;
using namespace base;





// **************** DevMesh ****************


DevMesh::DevMesh()
{
	mesh = NULL;
	adjacency = NULL;
	vdata = NULL;
	idata = NULL;
	adata = NULL;
}

DevMesh::DevMesh(const char *_path)
{
	mesh = NULL;
	adjacency = NULL;
	vdata = NULL;
	idata = NULL;
	adata = NULL;
	preload_path = _path;
}

DevMesh::~DevMesh()
{
	Clear(true);
}

bool DevMesh::Load(const char *path)
{
	if(mesh)
	{
		mesh->Release();
		mesh = NULL;
	}

	if(path[0] && path[strlen(path)-1]=='x')
	{
		HRESULT res = D3DXLoadMeshFromX(path,D3DXMESH_32BIT | D3DXMESH_MANAGED,
										Dev.GetDevice(),NULL,NULL,NULL,NULL,&mesh);
		return !FAILED(res) && mesh;
	}

	FileReaderStream file(path);
	TreeFileBuilder tfb;
	if(!tfb.LoadTreeBin(&file))
		return false;

	TreeFileRef root = tfb.GetRoot(false);
	TreeFileRef f_vb = root.SerChild("VertexBuffer");
	TreeFileRef f_ib = root.SerChild("IndexBuffer");
	TreeFileRef f_ab = root.SerChild("AttrBuffer");
	int vsize;
	f_vb.SerInt("VertexSize",vsize,3*sizeof(float));
	void  *vdata  = (void*)f_vb.Read_GetRawData("Data");
	int    vcount = f_vb.Read_GetRawSize("Data")/vsize;
	int   *idata  = (int*)f_ib.Read_GetRawData("Data");
	int    icount = f_ib.Read_GetRawSize("Data")/sizeof(int);
	DWORD *attr   = (DWORD*)f_ab.Read_GetRawData("Data");
	DWORD  acount = f_ab.Read_GetRawSize("Data")/sizeof(DWORD);
	if(acount*3 != icount)
		attr = NULL;

	DWORD FVF = D3DFVF_XYZ | D3DFVF_NORMAL | D3DFVF_TEX3 | D3DFVF_TEXCOORDSIZE3(1) | D3DFVF_TEXCOORDSIZE3(2);
	f_vb.SerDword("FVF",*(dword*)&FVF,FVF);

	vector<D3DXATTRIBUTERANGE> ranges;
	ranges.resize( f_ab.GetCloneArraySize("Range") );
	for(int i=0;i<(int)ranges.size();i++)
	{
		D3DXATTRIBUTERANGE *r = &ranges[i];
		TreeFileRef rf = f_ab.SerChild("Range",i);
		rf.SerDword("AttribId"		,*(dword*)&r->AttribId		,0);
		rf.SerDword("FaceStart"		,*(dword*)&r->FaceStart		,0);
		rf.SerDword("FaceCount"		,*(dword*)&r->FaceCount		,0);
		rf.SerDword("VertexStart"	,*(dword*)&r->VertexStart	,0);
		rf.SerDword("VertexCount"	,*(dword*)&r->VertexCount	,0);
	}

	bool ok = LoadVBIB(vdata,vcount,vsize,FVF,idata,icount,false,attr);
	if(!ok) return false;

	if(ranges.size()>0)
		ok = SUCCEEDED(mesh->SetAttributeTable(&ranges[0],(DWORD)ranges.size()));

	if(!ok)
		Clear(true);

	return ok;
}


bool DevMesh::LoadVBIB(const void *vdata,int vcount,int vsize,int FVF,int *idata,int icount,bool optimize,DWORD *attr)
{
	Clear(true);

	if(vsize<=0 || !vdata || vcount<=0 || !idata || icount<=0)
		return false;

	HRESULT res = D3D_OK;
	void *data;

	do {
		res = D3DXCreateMeshFVF(icount/3,vcount,D3DXMESH_MANAGED | D3DXMESH_32BIT,FVF,Dev.GetDevice(),&mesh);
		if(FAILED(res) || !mesh) break;

		// build vertexes
		res = mesh->LockVertexBuffer(0,&data);
		if(FAILED(res)) break;
		memcpy(data,vdata,vsize*vcount);
		res = mesh->UnlockVertexBuffer();
		if(FAILED(res)) break;

		// build indexes
		res = mesh->LockIndexBuffer(0,&data);
		if(FAILED(res)) break;
		memcpy(data,idata,4*icount);
		res = mesh->UnlockIndexBuffer();
		if(FAILED(res)) break;

		// build attributes
		res = mesh->LockAttributeBuffer(0,(DWORD**)&data);
		if(FAILED(res)) break;
		if(attr)	memcpy(data,attr,sizeof(DWORD)*icount/3);
		else		memset(data,0,sizeof(DWORD)*icount/3);
		res = mesh->UnlockAttributeBuffer();
		if(FAILED(res)) break;

		// optimize
		if(optimize)
		{
//			ID3DXMesh *m2;
//			res = mesh->CloneMeshFVF(D3DXMESH_MANAGED | D3DXMESH_32BIT,FVF,Dev.GetDevice(),&m2);
//			if(FAILED(res)) break;
//			mesh->Release();
//			mesh = m2;

			GenerateAdjacency(0.f);
			res = mesh->OptimizeInplace(
				D3DXMESHOPT_COMPACT | D3DXMESHOPT_ATTRSORT |
				D3DXMESHOPT_VERTEXCACHE | D3DXMESHOPT_DEVICEINDEPENDENT,
				adjacency,adjacency,NULL,NULL);
			Clear(false);
			
			if(FAILED(res)) break;
		}

	} while(0);

	if(FAILED(res))
	{
		Clear(true);
		return false;
	}

	return true;
}

bool DevMesh::Save(const char *path)
{
	if(mesh && FilePathGetPart(path,false,false,true)==".x")
		return SUCCEEDED( D3DXSaveMeshToX(path,mesh,0,0,0,0,D3DXF_FILEFORMAT_TEXT) );

	if(!ReadBack())
		return false;

	TreeFileBuilder tfb;
	TreeFileRef root = tfb.GetRoot(true);
	TreeFileRef f_vb = root.SerChild("VertexBuffer");
	TreeFileRef f_ib = root.SerChild("IndexBuffer");
	TreeFileRef f_ab = root.SerChild("AttrBuffer");

	// vertexes
	f_vb.SerInt("VertexSize",vsize,vsize);
	f_vb.Write_SetRaw("Data",vdata,vsize*vcount);
	int FVF = mesh->GetFVF();
	f_vb.SerInt("FVF",FVF,FVF);

	// indexes
	f_ib.Write_SetRaw("Data",idata,icount*sizeof(int));

	// attributes
	DWORD *abuffer = NULL;
	if(SUCCEEDED(mesh->LockAttributeBuffer(0,&abuffer)))
		f_ab.Write_SetRaw("Data",abuffer,mesh->GetNumFaces());

	vector<D3DXATTRIBUTERANGE> ranges;
	DWORD arcount = 0;
	if(SUCCEEDED(mesh->GetAttributeTable(NULL,&arcount)))
	{
		ranges.resize(arcount);
		if(ranges.size()>0 && SUCCEEDED(mesh->GetAttributeTable(&ranges[0],NULL)))
		{
			for(int i=0;i<(int)ranges.size();i++)
			{
				D3DXATTRIBUTERANGE *r = &ranges[i];
				TreeFileRef rf = f_ab.SerChild("Range",i);
				rf.SerDword("AttribId"		,*(dword*)&r->AttribId		,0);
				rf.SerDword("FaceStart"		,*(dword*)&r->FaceStart		,0);
				rf.SerDword("FaceCount"		,*(dword*)&r->FaceCount		,0);
				rf.SerDword("VertexStart"	,*(dword*)&r->VertexStart	,0);
				rf.SerDword("VertexCount"	,*(dword*)&r->VertexCount	,0);
			}
		}
	}

	// save
	FileWriterStream file(path);
	bool ok = tfb.SaveTreeBin(&file);

	if(abuffer)
		mesh->UnlockAttributeBuffer();

	Clear(false);

	return ok;
}

bool DevMesh::LoadCube(float w,float h,float d)
{
    Clear(true);

    return SUCCEEDED( D3DXCreateBox(Dev.GetDevice(),w,h,d,&mesh,NULL) );
}

bool DevMesh::LoadSphere(float r,int slices,int stacks)
{
    Clear(true);

    return SUCCEEDED( D3DXCreateSphere(Dev.GetDevice(),r,slices,stacks,&mesh,NULL) );
}

bool DevMesh::LoadCylinder(float r1,float r2,float len,int slices,int stacks)
{
    Clear(true);

    return SUCCEEDED( D3DXCreateCylinder(Dev.GetDevice(),r1,r2,len,slices,stacks,&mesh,NULL) );
}

bool DevMesh::LoadTorus(float rin,float rout,int sides,int rings)
{
    Clear(true);

    return SUCCEEDED( D3DXCreateTorus(Dev.GetDevice(),rin,rout,sides,rings,&mesh,NULL) );
}

bool DevMesh::LoadPolygon(float len,int sides)
{
    Clear(true);

    return SUCCEEDED( D3DXCreatePolygon(Dev.GetDevice(),len,sides,&mesh,NULL) );
}

void DevMesh::Clear(bool all)
{
	if(mesh && all)
	{
		mesh->Release();
		mesh = NULL;
	}
	if(adjacency) delete adjacency;
	if(vdata) delete vdata;
	if(idata) delete idata;
	if(adata) delete adata;
	adjacency = NULL;
	vdata = NULL;
	idata = NULL;
	adata = NULL;
}

void DevMesh::GenerateAdjacency(float epsilon)
{
	if(adjacency)
		delete adjacency;
	adjacency = NULL;
	if(!mesh || mesh->GetNumFaces()<=0)
		return;
	adjacency = new DWORD[3*mesh->GetNumFaces()];
	mesh->GenerateAdjacency(epsilon,adjacency);
}

void DevMesh::GenerateSoftAdjacency(float epsilon,float normal_dot,bool normal_dir)
{
	if(!mesh) return;
	if(normal_dot<=-1)
	{
		GenerateAdjacency(epsilon);
		return;
	}

	ReadBack();
	GenerateAdjacency(epsilon);

	for(int i=0;i<icount;i++)
		if(adjacency[i]>=0)
		{
			int base[2] = { (i/3) * 3, adjacency[i] * 3 };
			vec3 normal[2];

			for(int j=0;j<2;j++)
			{
				vec3 *v0 = (vec3*)(((byte*)vdata) + idata[base[j]  ]*vsize);
				vec3 *v1 = (vec3*)(((byte*)vdata) + idata[base[j]+1]*vsize);
				vec3 *v2 = (vec3*)(((byte*)vdata) + idata[base[j]+2]*vsize);
				normal[j] = (*v1-*v0).cross(*v2-*v0);
				normal[j].normalize();
				if(normal_dir)
					normal[j] *= -1;
			}

			float dot = normal[0].dot(normal[1]);
			if(dot<normal_dot)
				adjacency[i] = -1;
		}
}


bool DevMesh::ReadBack()
{
	Clear(false);
    if(!mesh) return false;

	HRESULT res = D3D_OK;
	void *p;
	vcount = mesh->GetNumVertices();
	vsize = mesh->GetNumBytesPerVertex();
	icount = mesh->GetNumFaces()*3;

	if(vcount<=0 || icount<=0)
		return false;

	do {
		vdata = new byte[vcount*vsize];
		res = mesh->LockVertexBuffer(0,(void**)&p);
		if(FAILED(res)) break;
		memcpy(vdata,p,vcount*vsize);
		res = mesh->UnlockVertexBuffer();
		if(FAILED(res)) break;

		idata = new int[icount];
		res = mesh->LockIndexBuffer(0,(void**)&p);
		if(FAILED(res)) break;
		memcpy(idata,p,icount*sizeof(int));
		res = mesh->UnlockIndexBuffer();
		if(FAILED(res)) break;

		adata = new DWORD[icount/3];
		res = mesh->LockAttributeBuffer(0,(DWORD**)&p);
		if(FAILED(res)) break;
		memcpy(adata,p,icount/3*sizeof(DWORD));
		res = mesh->UnlockAttributeBuffer();
		if(FAILED(res)) break;

	} while(0);

	if(FAILED(res))
	{
		Clear(false);
		return false;
	}

	return true;
}

bool DevMesh::Upload()
{
	HRESULT res = D3D_OK;
	void *p;

	if(!mesh || !vdata || !idata) return false;

	do {
		res = mesh->LockVertexBuffer(0,(void**)&p);
		if(FAILED(res)) break;
		memcpy(p,vdata,vcount*vsize);
		res = mesh->UnlockVertexBuffer();
		if(FAILED(res)) break;

		res = mesh->LockIndexBuffer(0,(void**)&p);
		if(FAILED(res)) break;
		memcpy(p,idata,icount*sizeof(int));
		res = mesh->UnlockIndexBuffer();
		if(FAILED(res)) break;

		if(adata)
		{
			res = mesh->LockAttributeBuffer(0,(DWORD**)&p);
			if(FAILED(res)) break;
			memcpy(p,adata,icount/3*sizeof(DWORD));
			res = mesh->UnlockAttributeBuffer();
			if(FAILED(res)) break;
		}
	} while(0);

	return SUCCEEDED(res);
}

void DevMesh::ApplyMatrixInMemory(D3DXMATRIX *mtx)
{
	if(!vdata) return;
	D3DXVec3TransformCoordArray(
		(D3DXVECTOR3*)vdata,vsize,
		(D3DXVECTOR3*)vdata,vsize,
		mtx,vcount);
}

void DevMesh::FlipTrisInMemory()
{
	if(!idata) return;
	for(int i=0;i<icount;i+=3)
	{
		int t = idata[i+1];
		idata[i+1] = idata[i+2];
		idata[i+2] = t;
	}
}

bool DevMesh::ComputeTangentFrame(tMeshAttrib tex_attr,tMeshAttrib normal_attr,
								tMeshAttrib tangent_attr,tMeshAttrib binormal_attr,
								float dot_normal,float dot_tangent,float eps_singular,int options)
{
	if(!mesh) return false;

	options |= D3DXTANGENT_WEIGHT_BY_AREA;// | D3DXTANGENT_GENERATE_IN_PLACE;
	if(!ATTR_IS_PRESENT(tex_attr)) options |= D3DXTANGENT_CALCULATE_NORMALS;

	ID3DXMesh *new_mesh = NULL;
	HRESULT res = D3DXComputeTangentFrameEx(
		mesh,														// mesh
		ATTR_GET_USAGE(tex_attr),		ATTR_GET_ID(tex_attr),		// input texcoord
		ATTR_GET_USAGE(tangent_attr),	ATTR_GET_ID(tangent_attr),	// out tangent
		ATTR_GET_USAGE(binormal_attr),	ATTR_GET_ID(binormal_attr),	// out binormal
		ATTR_GET_USAGE(normal_attr),	ATTR_GET_ID(normal_attr),	// out normal
		options,						// options
		adjacency,						// adjacency
		dot_tangent,					// tangent/binormal merge treshold
		eps_singular,					// singular point treshold
		dot_normal,						// normal threshold
		&new_mesh,						// mesh out
		NULL							// vertex remap out
		);

	assert(res!=D3DERR_INVALIDCALL);
	assert(res!=D3DXERR_INVALIDDATA);
	assert(res!=E_OUTOFMEMORY);

	if(new_mesh)
	{
		mesh->Release();
		mesh = new_mesh;
	}

	Clear(false);

	return SUCCEEDED(res);
}

void DevMesh::GenerateDegenerateEdges(int flags,tMeshAttrib normal_attr,DevMesh *out)
{
	if(!mesh) return;

	ReadBack();
	GenerateAdjacency(0);

	vector<byte> nvdata;
	nvdata.resize(icount*vsize);

	for(int i=0;i<icount;i++)
		memcpy(&nvdata[i*vsize],&((byte*)vdata)[idata[i]*vsize],vsize);

	if(ATTR_IS_PRESENT(normal_attr))
	{
		D3DVERTEXELEMENT9 decl[MAX_FVF_DECL_SIZE];
		mesh->GetDeclaration(decl);

		int offs = -1;
		for(int i=0;decl[i].Stream!=0xFF;i++)
			if(decl[i].Usage==ATTR_GET_USAGE(normal_attr) && decl[i].UsageIndex==ATTR_GET_ID(normal_attr))
				offs = decl[i].Offset;

		if(offs>=0)
		{
			for(int i=0;i<icount;i+=3)
			{
				D3DXVECTOR3 d1 = *(D3DXVECTOR3*)&nvdata[(i+1)*vsize] - *(D3DXVECTOR3*)&nvdata[i*vsize];
				D3DXVECTOR3 d2 = *(D3DXVECTOR3*)&nvdata[(i+2)*vsize] - *(D3DXVECTOR3*)&nvdata[i*vsize];
				D3DXVECTOR3 n;
				D3DXVec3Cross(&n,&d1,&d2);
				D3DXVec3Normalize(&n,&n);
				*(D3DXVECTOR3*)&nvdata[ i   *vsize+offs] = n;
				*(D3DXVECTOR3*)&nvdata[(i+1)*vsize+offs] = n;
				*(D3DXVECTOR3*)&nvdata[(i+2)*vsize+offs] = n;
			}
		}
	}


	vector<int> nidata;
	nidata.resize(icount);

	for(int i=0;i<icount;i++)
		nidata[i] = i;

	for(int i=0;i<icount;i++)
	{
		if(adjacency[i]==0xFFFFFFFF) continue;
//		if(adjacency[i]<=i) continue;
		int j = -1;
		for(int k=0;k<3;k++)
			if(adjacency[adjacency[i]*3+k]==i/3)
				j = adjacency[i]*3+k;
		if(j<0) continue;
		int a0 = i, b0 = j;
		int a1 = a0 - (a0%3) + ((a0+1)%3);
		int b1 = b0 - (b0%3) + ((b0+1)%3);
		nidata.push_back(a0);
		nidata.push_back(b1);
		nidata.push_back(a1);
		nidata.push_back(a1);
		nidata.push_back(b1);
		nidata.push_back(b0);
	}

	out->LoadVBIB(&nvdata[0],(int)nvdata.size()/vsize,vsize,mesh->GetFVF(),&nidata[0],(int)nidata.size());
}


static int _decltype_to_size(int decl)
{
	if(decl==D3DDECLTYPE_FLOAT1) return sizeof(float);
	if(decl==D3DDECLTYPE_FLOAT2) return 2*sizeof(float);
	if(decl==D3DDECLTYPE_FLOAT3) return 3*sizeof(float);
	if(decl==D3DDECLTYPE_FLOAT4) return 4*sizeof(float);
	if(decl==D3DDECLTYPE_D3DCOLOR) return sizeof(DWORD);
	return 0;
}

bool DevMesh::ReorderVertexFields(int FVF,tMeshAttrib attr_map[][2])
{
	if(!mesh) return false;

	ReadBack();

	HRESULT res = D3D_OK;
	void *data;
	ID3DXMesh *m2 = NULL;

	do {
		res = D3DXCreateMeshFVF(icount/3,vcount,D3DXMESH_MANAGED | D3DXMESH_32BIT,FVF,Dev.GetDevice(),&m2);
		if(FAILED(res)) break;

		// read declarations
		D3DVERTEXELEMENT9 srcd[MAX_FVF_DECL_SIZE], dstd[MAX_FVF_DECL_SIZE];
		mesh->GetDeclaration(srcd);
		m2->GetDeclaration(dstd);

		// build copy table
		int nvsize = m2->GetNumBytesPerVertex();
		int copy_offs[1024];
		memset(copy_offs,-1,sizeof(copy_offs));

		if(attr_map)
		{
			for(int i=0;ATTR_IS_PRESENT(attr_map[i][0]);i++)
			{
				int doffs=-1,soffs=-1,size=0;
				for(int j=0;dstd[j].Stream!=0xFF;j++)
					if(dstd[j].Usage==ATTR_GET_USAGE(attr_map[i][0]) && dstd[j].UsageIndex==ATTR_GET_ID(attr_map[i][0]))
					{
						doffs = dstd[j].Offset;
						size = _decltype_to_size(dstd[j].Type);
					}
				for(int j=0;srcd[j].Stream!=0xFF;j++)
					if(srcd[j].Usage==ATTR_GET_USAGE(attr_map[i][1]) && srcd[j].UsageIndex==ATTR_GET_ID(attr_map[i][0]))
						soffs = srcd[j].Offset;
				if(doffs>=0 && soffs>=0 && size>0)
					for(int j=0;j<size/4;j++)
						copy_offs[doffs/4+j] = soffs/4+j;
			}
		}
		else
		{
			for(int i=0;dstd[i].Stream!=0xFF;i++)
			{
				int doffs=-1,soffs=-1,size=0;
				for(int j=0;srcd[j].Stream!=0xFF;j++)
					if(dstd[i].Usage==srcd[j].Usage && dstd[i].UsageIndex==srcd[i].UsageIndex)
					{
						doffs = dstd[j].Offset;
						soffs = srcd[j].Offset;
						size = _decltype_to_size(dstd[j].Type);
						break;
					}

				if(doffs>=0 && soffs>=0 && size>0)
					for(int j=0;j<size/4;j++)
						copy_offs[doffs/4+j] = soffs/4+j;
			}
		}

		// build vertexes
		res = m2->LockVertexBuffer(0,&data);
		if(FAILED(res)) break;
		for(int i=0;i<vcount;i++)
			for(int j=0;j<nvsize/4;j++)
				if(copy_offs[j]>=0)
					*(((DWORD*)data)+(i*nvsize/4+j)) = *(((DWORD*)vdata)+(i*vsize/4+copy_offs[j]));
				else
					*(((DWORD*)data)+(i*nvsize/4+j)) = 0;
		res = m2->UnlockVertexBuffer();
		if(FAILED(res)) break;

		// build indexes
		res = m2->LockIndexBuffer(0,&data);
		if(FAILED(res)) break;
		memcpy(data,idata,4*icount);
		res = m2->UnlockIndexBuffer();
		if(FAILED(res)) break;

		// build attributes
		res = m2->LockAttributeBuffer(0,(DWORD**)&data);
		if(FAILED(res)) break;
		memset(data,0,sizeof(DWORD)*icount/3);
		res = m2->UnlockAttributeBuffer();
		if(FAILED(res)) break;
	} while(0);

	if(FAILED(res))
	{
		if(m2) m2->Release();
		m2 = NULL;
		return false;
	}

	Clear(true);
	mesh = m2;

	return true;
}


bool DevMesh::UnwrapUV(float max_stretch,int tex_w,int tex_h,float gutter,int tex_id,float normal_dot,bool normal_dir)
{
	if(!mesh) return false;

	ID3DXMesh *new_mesh = NULL;

	GenerateSoftAdjacency(0.f,normal_dot,normal_dir);

	HRESULT res = D3DXUVAtlasCreate(
		mesh,					// mesh
		0,						// max charts
		max_stretch,			// max stretch
		tex_w, tex_h,			// texture size
		gutter,					// gutter
		tex_id,					// texture index
		adjacency,				// adjacency
		NULL,					// false edges
		NULL,					// IMT array
		NULL,					// callback
		0,						// callback frequency
		NULL,					// callback param
		D3DXUVATLAS_DEFAULT,	// options
		&new_mesh,				// out mesh
		NULL,					// face partitioning
		NULL,					// vertex remap
		NULL,					// max stretch out
		NULL					// num charts out
	);

	assert(res!=D3DERR_INVALIDCALL);
	assert(res!=D3DXERR_INVALIDDATA);
	assert(res!=E_OUTOFMEMORY);

	if(SUCCEEDED(res) && new_mesh)
	{
		mesh->Release();
		mesh = new_mesh;
	}

	Clear(false);

	return SUCCEEDED(res);
}

bool DevMesh::CleanMesh()
{
	if(GetIndexCount()<3)
		return true;

	ID3DXMesh *new_mesh = NULL;

	GenerateAdjacency(0);

	ID3DXBuffer *err = NULL;
	HRESULT res = D3DXCleanMesh(
		D3DXCLEAN_SIMPLIFICATION,	// flags
		mesh,						// mesh
		adjacency,					// adjacency
		&new_mesh,					// new mesh
		adjacency,					// out adjacency
		&err						// errors
	);

	assert(res!=D3DERR_INVALIDCALL);
	assert(res!=D3DXERR_INVALIDDATA);
	assert(res!=E_OUTOFMEMORY);

	if(err)
		err->Release();

	if(SUCCEEDED(res) && new_mesh)
	{
		mesh->Release();
		mesh = new_mesh;
	}

	Clear(false);

	return SUCCEEDED(res);
}

bool DevMesh::Optimize()
{
	if(!mesh) return true;
	GenerateAdjacency(0);
	return SUCCEEDED(mesh->OptimizeInplace(
				D3DXMESHOPT_COMPACT | D3DXMESHOPT_ATTRSORT | D3DXMESHOPT_VERTEXCACHE | D3DXMESHOPT_DONOTSPLIT,
				adjacency,adjacency,NULL,NULL));
}

bool DevMesh::Simplify(int size,bool count_faces)
{
	if(GetIndexCount()<3)
		return true;

	ID3DXMesh *new_mesh = NULL;

	GenerateAdjacency(0);

	HRESULT res = D3DXSimplifyMesh(
		mesh,													// mesh
		adjacency,												// adjacency
		NULL,													// attribute weights
		NULL,													// vertex weights
		size,													// min value
		count_faces ? D3DXMESHSIMP_FACE : D3DXMESHSIMP_VERTEX,	// options
		&new_mesh												// new mesh
	);

	assert(res!=D3DERR_INVALIDCALL);
	assert(res!=D3DXERR_INVALIDDATA);
	assert(res!=E_OUTOFMEMORY);

	if(SUCCEEDED(res) && new_mesh)
	{
		mesh->Release();
		mesh = new_mesh;
	}

	Clear(false);

	return SUCCEEDED(res);
}

int DevMesh::GetVertexStride()
{
	if(!mesh) return 0;
	return mesh->GetNumBytesPerVertex();
}

int	DevMesh::GetVertexCount()
{
	if(!mesh) return 0;
	return mesh->GetNumVertices();
}

bool DevMesh::CopyVertexData(void *buffer)
{
	void *data;
	if(!mesh) return false;
	HRESULT res = mesh->LockVertexBuffer(0,&data);
	if(FAILED(res)) return false;
	memcpy(buffer,data,GetVertexDataSize());
	res = mesh->UnlockVertexBuffer();
	if(FAILED(res)) return false;
	return true;
}

int DevMesh::GetIndexCount()
{
	if(!mesh) return 0;
	return mesh->GetNumFaces()*3;
}

bool DevMesh::CopyIndexData(int *buffer)
{
	void *data;
	if(!mesh) return false;
	HRESULT res = mesh->LockIndexBuffer(0,&data);
	if(FAILED(res)) return false;
	memcpy(buffer,data,4*GetIndexCount());
	res = mesh->UnlockIndexBuffer();
	if(FAILED(res)) return false;
	return true;
}

int DevMesh::GetRangeCount()
{
	if(!mesh) return 0;
	DWORD size = 0;
	if(FAILED(mesh->GetAttributeTable(NULL,&size)))
		return 0;
	return size;
}

bool DevMesh::CopyRangeData(DevMesh::Range *buffer)
{
	if(!mesh) return false;
	return SUCCEEDED(mesh->GetAttributeTable((D3DXATTRIBUTERANGE*)buffer,NULL));
}

void DevMesh::DrawSection(int id)
{
	if(mesh)
		mesh->DrawSubset(id);
}

void DevMesh::DrawRange(const Range &r)
{
	if(!mesh) return;
	DWORD fvf = mesh->GetFVF();
	IDirect3DVertexBuffer9 *vb = NULL;
	IDirect3DIndexBuffer9 *ib = NULL;
	mesh->GetVertexBuffer(&vb);
	mesh->GetIndexBuffer(&ib);

	if(fvf && vb && ib)
	{
		Dev->SetFVF(fvf);
		Dev->SetStreamSource(0,vb,0,mesh->GetNumBytesPerVertex());
		Dev->SetIndices(ib);
		Dev->DrawIndexedPrimitive(D3DPT_TRIANGLELIST,0,r.vtx_start,r.vtx_count,r.face_start*3,r.face_count);
	}

	if(vb) vb->Release();
	if(ib) ib->Release();
}

void DevMesh::DrawTriangleRange(int first_tri,int num_tris)
{
	if(!mesh) return;
	DWORD fvf = mesh->GetFVF();
	IDirect3DVertexBuffer9 *vb = NULL;
	IDirect3DIndexBuffer9 *ib = NULL;
	mesh->GetVertexBuffer(&vb);
	mesh->GetIndexBuffer(&ib);

	if(fvf && vb && ib)
	{
		Dev->SetFVF(fvf);
		Dev->SetStreamSource(0,vb,0,mesh->GetNumBytesPerVertex());
		Dev->SetIndices(ib);
		Dev->DrawIndexedPrimitive(D3DPT_TRIANGLELIST,0,0,mesh->GetNumVertices(),first_tri*3,num_tris);
	}

	if(vb) vb->Release();
	if(ib) ib->Release();
}

void DevMesh::OnBeforeReset()
{
}

void DevMesh::OnAfterReset()
{
}

void DevMesh::OnPreload()
{
	if(preload_path.size()>0)
		Load(preload_path.c_str());
}

/*
void DevMesh::MakeDeclFromFVFCode(int FVF,int &decl,int &id)
{
	if(FVF==D3DFVF_XYZ		) { decl = D3DDECLUSAGE_POSITION;	id = 0; return; }
	if(FVF==D3DFVF_NORMAL	) { decl = D3DDECLUSAGE_NORMAL;		id = 0;	return; }
	if(FVF==D3DFVF_PSIZE	) { decl = D3DDECLUSAGE_PSIZE;		id = 0;	return; }
	if(FVF==D3DFVF_TEX1		) { decl = D3DDECLUSAGE_TEXCOORD;	id = 0;	return; }
	if(FVF==D3DFVF_TEX2		) { decl = D3DDECLUSAGE_TEXCOORD;	id = 1;	return; }
	if(FVF==D3DFVF_TEX3		) { decl = D3DDECLUSAGE_TEXCOORD;	id = 2;	return; }
	if(FVF==D3DFVF_TEX4		) { decl = D3DDECLUSAGE_TEXCOORD;	id = 3;	return; }
	if(FVF==D3DFVF_TEX5		) { decl = D3DDECLUSAGE_TEXCOORD;	id = 4;	return; }
	if(FVF==D3DFVF_TEX6		) { decl = D3DDECLUSAGE_TEXCOORD;	id = 5;	return; }
	if(FVF==D3DFVF_TEX7		) { decl = D3DDECLUSAGE_TEXCOORD;	id = 6;	return; }
	if(FVF==D3DFVF_TEX8		) { decl = D3DDECLUSAGE_TEXCOORD;	id = 7;	return; }
	decl = D3DX_DEFAULT;
	id = 0;
}
*/

// ******************************** d_resources.cpp ********************************

// ---- #include "dxfw.h"
// ---> including dxfw.h
// <--- back to d_resources.cpp

using namespace std;
using namespace base;




// **************** DevRenderTarget ****************


DevRenderTarget::DevRenderTarget()
{
	rt = NULL;
	format = D3DFMT_A8R8G8B8;
	width = 0;
	height = 0;
	denominator = 1;
	flags = 0;
}

DevRenderTarget::DevRenderTarget(int _format,int _width,int _height,int _denominator,int _flags)
{
	rt = NULL;
	format = _format;
	width = _width;
	height = _height;
	denominator = _denominator;
	flags = _flags;
}

DevRenderTarget::~DevRenderTarget()
{
	if(rt) rt->Release();
	rt = NULL;
}

IDirect3DSurface9 *DevRenderTarget::GetSurface()
{
	int w, h, m=1;
	GetCurrentSize(w,h);
	if(!rt)
	{
		DWORD n_usage = 0;
		if(!Device::IsFormatDepthStencil((D3DFORMAT)format))
			n_usage |= D3DUSAGE_RENDERTARGET;

		if(flags & (TEXF_MIPMAPS | TEXF_AUTO_MIPMAPS))
			m = 0;

		if(flags & TEXF_AUTO_MIPMAPS)
			n_usage |= D3DUSAGE_AUTOGENMIPMAP;

		HRESULT res = Dev->CreateTexture(w,h,m,n_usage,(D3DFORMAT)format,D3DPOOL_DEFAULT,&rt,NULL);
		if(FAILED(res) || !rt) return NULL;
	}

	IDirect3DSurface9 *surf = NULL;
	rt->GetSurfaceLevel(0,&surf);
	return surf;
}

void DevRenderTarget::SetParameters(int _format,int _width,int _height,int _denominator,int _flags)
{
	if(format==_format && width==_width && height==_height && denominator==_denominator && flags==_flags)
		return;

	if(rt) rt->Release();
	rt = NULL;

	format = _format;
	width = _width;
	height = _height;
	denominator = _denominator;
	flags = _flags;
}

void DevRenderTarget::GetCurrentSize(int &sx,int &sy)
{
	if(denominator>0)
	{
		int dsx, dsy;
		Dev.GetScreenSize(dsx,dsy);
		sx = (dsx + (denominator-1))/denominator;
		sy = (dsy + (denominator-1))/denominator;
	}
	else
	{
		sx = width;
		sy = height;
	}
}

vec2 DevRenderTarget::GetCurrentSizeV()
{
	int sx, sy;
	GetCurrentSize(sx,sy);
	return vec2((float)sx,(float)sy);
}

bool DevRenderTarget::Save(const char *path)
{
	IDirect3DSurface9 *surf = GetSurface();
	if(!surf) return false;

	D3DXSaveSurfaceToFile(path,D3DXIFF_PNG,surf,NULL,NULL);
	surf->Release();

	return true;
}

void DevRenderTarget::OnBeforeReset()
{
	if(rt) rt->Release();
	rt = NULL;
}

void DevRenderTarget::OnAfterReset()
{
}


// **************** DevCubeRenderTarget ****************


DevCubeRenderTarget::DevCubeRenderTarget(int _format,int _size)
{
	rt = NULL;
	format = _format;
	size = _size;
}

DevCubeRenderTarget::~DevCubeRenderTarget()
{
	if(rt) rt->Release();
	rt = NULL;
}

IDirect3DSurface9 *DevCubeRenderTarget::GetSurface(int face)
{
	if(!rt)
	{
		DWORD n_usage = 0;
		if(!Device::IsFormatDepthStencil((D3DFORMAT)format))
			n_usage |= D3DUSAGE_RENDERTARGET;
		HRESULT res = Dev->CreateCubeTexture(size,1,n_usage,(D3DFORMAT)format,D3DPOOL_DEFAULT,&rt,NULL);
		if(FAILED(res) || !rt) return NULL;
	}

	IDirect3DSurface9 *surf = NULL;
	rt->GetCubeMapSurface((D3DCUBEMAP_FACES)face,0,&surf);
	return surf;
}

void DevCubeRenderTarget::SetParameters(int _format,int _size)
{
	if(format==_format && size==_size)
		return;

	if(rt) rt->Release();
	rt = NULL;

	format = _format;
	size = _size;
}

void DevCubeRenderTarget::OnBeforeReset()
{
	if(rt) rt->Release();
	rt = NULL;
}

void DevCubeRenderTarget::OnAfterReset()
{
}



// **************** DevDepthStencilSurface ****************


DevDepthStencilSurface::DevDepthStencilSurface(int _format,int _width,int _height,int _denominator)
{
	surf = NULL;
	format = _format;
	width = _width;
	height = _height;
	denominator = _denominator;
}

DevDepthStencilSurface::~DevDepthStencilSurface()
{
	if(surf) surf->Release();
	surf = NULL;
}

IDirect3DSurface9 *DevDepthStencilSurface::GetSurface()
{
	int w, h;
	GetCurrentSize(w,h);
	if(!surf)
	{
		HRESULT res = Dev->CreateDepthStencilSurface(w,h,(D3DFORMAT)format,D3DMULTISAMPLE_NONE,0,false,&surf,NULL);
		if(FAILED(res) || !surf)
		{
//			MessageBox(NULL,base::format("Failed to create secondary z/stencil surface (code %08x)",res).c_str(),"Error!",MB_OK);
			return NULL;
		}
	}
	return surf;
}

void DevDepthStencilSurface::GetCurrentSize(int &sx,int &sy)
{
	if(denominator>0)
	{
		int dsx, dsy;
		Dev.GetScreenSize(dsx,dsy);
		sx = (dsx + (denominator-1))/denominator;
		sy = (dsy + (denominator-1))/denominator;
	}
	else
	{
		sx = width;
		sy = height;
	}
}

void DevDepthStencilSurface::OnBeforeReset()
{
	if(surf) surf->Release();
	surf = NULL;
}

void DevDepthStencilSurface::OnAfterReset()
{
}




// **************** DevFont ****************

DevFont::DevFont()
{
	font = NULL;
	do_preload = false;
}

DevFont::DevFont(const char *facename,int height,int width,bool bold,bool italic)
{
	font = NULL;
	do_preload = false;
	if(Dev.GetIsReady())
		Create(facename,height,width,bold,italic);
	else
	{
		do_preload = true;
		p_facename = facename;
		p_height = height;
		p_width = width;
		p_bold = bold;
		p_italic = italic;
	}
}

DevFont::~DevFont()
{
	if(font)
		font->Release();
}

bool DevFont::Create(const char *facename,int height,int width,bool bold,bool italic)
{
	if(font)
		font->Release();
	font = NULL;
	do_preload = false;

	HRESULT res = D3DXCreateFont(Dev.GetDevice(),height,width,bold?700:400,3,italic,
									DEFAULT_CHARSET,OUT_DEFAULT_PRECIS,ANTIALIASED_QUALITY,
									FF_DONTCARE,facename,&font);
	return SUCCEEDED(res);
}

void DevFont::OnBeforeReset()
{
	if(font)
		font->OnLostDevice();
}

void DevFont::OnAfterReset()
{
	if(font)
		font->OnResetDevice();
}

// ******************************** d_shaderset.cpp ********************************

// ---- #include "dxfw.h"
// ---> including dxfw.h
// <--- back to d_shaderset.cpp

using namespace std;
using namespace base;



// ----------------


DevShaderSet::DevShaderSet(const char *path,DefGenerator *dg,const char *v)
{
	def_generator = dg;
	version = v;

	if(Dev.GetIsReady())	Load(path,dg);
	else					p_path = path;
}

DevShaderSet::~DevShaderSet()
{
	ClearAll();
}

bool DevShaderSet::Load(const char *path,DefGenerator *dg)
{
	// clear
	ClearAll();

	// load file
	if(!ScrambleLoad(path,source))
		return false;
	original_path = path;

	// build defines
	if(dg)	dg(defs);
	else	defs.push_back(":");

	// shatter defs
	for(int i=0;i<(int)defs.size();i++)
	{
		macro_start.push_back((DWORD)macro_ptrs.size());
		ShatterDefsList(&defs[i][0],macro_ptrs);
	}

	// compile
	for(int i=0;i<(int)defs.size();i++)
		CompileVersion(i,false);

	return false;
}

bool DevShaderSet::BindVersionStr(const char *version)
{
	map<string,int>::iterator p;
	p = vmap.find(version);
	if( p==vmap.end() )
		return false;

	return BindVersion(p->second);
}

bool DevShaderSet::BindVersion(int version)
{
	if( version<0 || version>=(int)versions.size() )
		return false;

	ShaderSet *v = &versions[version];
	if( !v->vshader || !v->pshader )
		CompileVersion( version, true );

	Dev->SetVertexShader( v->vshader );
	Dev->SetPixelShader( v->pshader );

	return true;
}

void DevShaderSet::Unbind()
{
	Dev->SetVertexShader( NULL );
	Dev->SetPixelShader( NULL );
}

int	DevShaderSet::GetVersionIndex(const char *version)
{
	map<string,int>::iterator p;
	p = vmap.find(version);
	if( p==vmap.end() )
		return -1;

	return p->second;
}


void DevShaderSet::OnBeforeReset()
{
	ReleaseAll();
}

void DevShaderSet::OnAfterReset()
{
}

void DevShaderSet::ClearAll()
{
	ReleaseAll();

	source.clear();
	defs.clear();
	macro_ptrs.clear();
	macro_start.clear();
	original_path.clear();

	raw_file.clear();
	versions.clear();
	vmap.clear();
}

void DevShaderSet::ReleaseAll()
{
	for(int i=0;i<(int)versions.size();i++)
	{
		if(versions[i].vshader) versions[i].vshader->Release();
		if(versions[i].pshader) versions[i].pshader->Release();
		versions[i].vshader = NULL;
		versions[i].pshader = NULL;
	}
}

bool DevShaderSet::ScrambleLoad(const char *path,vector<char> &data)
{
	bool scrambled = false;
	if(!NFS.GetFileBytes(path,*(vector<byte>*)&data) || data.size()==0)
	{
		if(MessageBox(NULL,format("Can't find file %s!",path).c_str(),path,MB_OKCANCEL)==IDCANCEL)
			ExitProcess(0);
		return false;
	}

	data.push_back(0);

	return true;
}

void DevShaderSet::ShatterDefsList(char *s,vector<D3DXMACRO> &macros)
{
	// parse version identifier
	const char *id = s;
	while(*s && *s!=':') s++;

	// parse macros
	while(*s)
	{
		D3DXMACRO m = { NULL, NULL };

		// terminate previous string
		*s++ = 0;
		
		// read name
		m.Name = s;
		while(*s && *s!='=') s++;
		if(!*s) break;
		*s++ = 0;

		// read definition
		m.Definition = s;
		while(*s && *s!=';') s++;

		macros.push_back(m);
	}

	// terminate macro list
	D3DXMACRO m = { NULL, NULL };
	macros.push_back(m);
}

void DevShaderSet::CreateShaderCache( vector<string> &defs, byte hash[16] )
{
	raw_file.clear();
	versions.clear();
	vmap.clear();

	int nv = (int)defs.size();
	versions.resize(nv);

	raw_file.assign( hash, hash+16 );
	raw_file.insert( raw_file.end(), (byte*)&nv, (byte*)(&nv+1) );
	if(nv>0)
		raw_file.insert( raw_file.end(), (byte*)&versions[0], (byte*)(&versions[0] + versions.size()) );

	for(int i=0;i<(int)versions.size();i++)
	{
		const char *s = defs[i].c_str();

		versions[i].name_ofs = (int)raw_file.size();
		while(*s && *s!=':')
			raw_file.push_back(*s++);
		raw_file.push_back(0);

		versions[i].vcode_ofs = 0;
		versions[i].pcode_ofs = 0;
		versions[i].vshader = NULL;
		versions[i].pshader = NULL;

		vmap[ (char *)&raw_file[versions[i].name_ofs] ] = i;
	}
}

bool DevShaderSet::LoadShaderCache( const char *path, byte hash[16] )
{
	raw_file.clear();
	versions.clear();
	vmap.clear();

	return false;
}

bool DevShaderSet::SaveShaderCache( const char *path )
{
	if(versions.size()>=1)
		memcpy( &raw_file[16+sizeof(int)], &versions[0], sizeof(ShaderSet)*versions.size() );

	return true;
}

void DevShaderSet::CompileVersion( int ver, bool create_shaders )
{
	if( ver<0 || ver>=(int)versions.size() || source.size()<=0 )
		return;

	ShaderSet *v = &versions[ver];
	char profile[20];

	for(int type=0;type<2;type++)
	{
		ID3DXBuffer *buffer = NULL, *errors = NULL;
		HRESULT res;
		DWORD *code_ofs = type ? &v->pcode_ofs : &v->vcode_ofs;
		const char *entry = type ? "pmain" : "vmain";
		sprintf_s(profile,20,"%s_%s",type?"ps":"vs",version);

		if(*code_ofs != 0)
			continue;

		res = D3DXCompileShader(
					&source[0], (DWORD)source.size(),
					&macro_ptrs[macro_start[ver]], NULL, entry, profile,
					D3DXSHADER_AVOID_FLOW_CONTROL, &buffer, &errors, NULL );
		if( SUCCEEDED(res) && buffer )
		{
			byte *bp = (byte*) buffer->GetBufferPointer();

			*code_ofs = (DWORD)raw_file.size();
			raw_file.insert( raw_file.end(), bp, bp + buffer->GetBufferSize() );
		}
		else
		{
			const char *err = "Unknown error!";
			if(errors)
				err = (char*)errors->GetBufferPointer();
			
			const char *def = "?";
			if(ver<(int)defs.size()) def = defs[ver].c_str();
			if(MessageBox(NULL,err,format("%s:%s:%cs",original_path.c_str(),def,"vp"[type]).c_str(),MB_OKCANCEL)==IDCANCEL)
				ExitProcess(0);

			*code_ofs = 0xFFFFFFFF;
		}

		if(buffer) buffer->Release();
		if(errors) errors->Release();
	}

	if( create_shaders )
	{
		if(v->vcode_ofs != 0xFFFFFFFF && v->pshader==NULL)
		{
			if(FAILED(Dev->CreateVertexShader((DWORD*)&raw_file[v->vcode_ofs],&v->vshader)))
			{
				const char *def = "?";
				if(ver<(int)defs.size()) def = defs[ver].c_str();
				if(MessageBox(NULL,"Internal error!",format("%s:%s:%vs",original_path.c_str(),def).c_str(),MB_OKCANCEL)==IDCANCEL)
					ExitProcess(0);
				v->vcode_ofs = 0xFFFFFFFF;
			}
		}

		if(v->pcode_ofs != 0xFFFFFFFF && v->pshader==NULL)
		{
			if(FAILED(Dev->CreatePixelShader((DWORD*)&raw_file[v->pcode_ofs],&v->pshader)))
			{
				const char *def = "?";
				if(ver<(int)defs.size()) def = defs[ver].c_str();
				if(MessageBox(NULL,"Internal error!",format("%s:%s:%ps",original_path.c_str(),def).c_str(),MB_OKCANCEL)==IDCANCEL)
					ExitProcess(0);
				v->pcode_ofs = 0xFFFFFFFF;
			}
		}
	}
}

// ******************************** d_texture.cpp ********************************

// ---- #include "dxfw.h"
// ---> including dxfw.h
// <--- back to d_texture.cpp

using namespace std;
using namespace base;




// **************** DevTexture ****************


DevTexture::DevTexture()
{
	tex = NULL;
}

DevTexture::DevTexture(const char *_path)
{
	tex = NULL;
	preload_path = _path;
}

DevTexture::~DevTexture()
{
	if(tex)
		tex->Release();
}

void DevTexture::Unload()
{
	if(tex)
	{
		tex->Release();
		tex = NULL;
	}
}

bool DevTexture::Load(const char *path)
{
	Unload();

	static const char *EXT[] = {
		"",".png",".jpg",".tga",".bmp",".dds",".hdr",NULL
	};

	HRESULT res;
	string spath;
	for(int i=0;EXT[i];i++)
	{
		spath = format("%s%s",path,EXT[i]);

		res = D3DXCreateTextureFromFileEx(
			Dev.GetDevice(),
			spath.c_str(),
			D3DX_DEFAULT_NONPOW2, D3DX_DEFAULT_NONPOW2,
			0, 0,
			D3DFMT_A8R8G8B8, D3DPOOL_MANAGED,
			D3DX_DEFAULT, D3DX_DEFAULT, 0, 0, 0, (IDirect3DTexture9**)&tex);

		if(SUCCEEDED(res) && tex)
		{
			path = spath.c_str();
			break;
		}
	}

	if(FAILED(res) || !tex)
		return false;

	IDirect3DTexture9 *ctex = (IDirect3DTexture9*)tex;
	D3DSURFACE_DESC cd, ad, dd;
	res = ctex->GetLevelDesc(0,&cd);
	if(FAILED(res) || (cd.Format!=D3DFMT_A8R8G8B8 && cd.Format!=D3DFMT_X8R8G8B8))
		return true;

	string path_a = FilePathGetPart(path,true,true,false) + "_a" + FilePathGetPart(path,false,false,true);
	FILE *fp = NULL;
	fopen_s(&fp,path_a.c_str(),"rb");
	if(!fp) return true;
	fclose(fp);

	IDirect3DTexture9 *atex = NULL;
	res = D3DXCreateTextureFromFileEx(
			Dev.GetDevice(),path_a.c_str(),D3DX_DEFAULT_NONPOW2,D3DX_DEFAULT_NONPOW2,0,0,D3DFMT_X8R8G8B8,D3DPOOL_MANAGED,
			D3DX_DEFAULT, D3DX_DEFAULT, 0, 0, 0, (IDirect3DTexture9**)&atex);

	if(FAILED(res) || !atex)
		return true;

	IDirect3DTexture9 *dtex = NULL;
	bool all_ok = false;
	do {
		if(ctex->GetLevelCount() != atex->GetLevelCount())
			break;

		res = Dev->CreateTexture(cd.Width,cd.Height,ctex->GetLevelCount(),cd.Usage,D3DFMT_A8R8G8B8,cd.Pool,&dtex,NULL);
		if(FAILED(res) || !dtex) break;

		if(dtex->GetLevelCount() != ctex->GetLevelCount())
			break;

		for(int i=0;i<(int)ctex->GetLevelCount();i++)
		{
			if(FAILED( ctex->GetLevelDesc(i,&cd) )) break;
			if(FAILED( atex->GetLevelDesc(i,&ad) )) break;
			if(FAILED( dtex->GetLevelDesc(i,&dd) )) break;

			if(cd.Format!=D3DFMT_A8R8G8B8 && cd.Format!=D3DFMT_X8R8G8B8) break;
			if(ad.Format!=D3DFMT_A8R8G8B8 && ad.Format!=D3DFMT_X8R8G8B8) break;
			if(dd.Format!=D3DFMT_A8R8G8B8) break;
			if(cd.Width  != ad.Width ) break;
			if(cd.Height != ad.Height) break;
			if(cd.Width  != dd.Width ) break;
			if(cd.Height != dd.Height) break;

			D3DLOCKED_RECT clr, alr, dlr;
			if(FAILED( atex->LockRect(i,&alr,NULL,D3DLOCK_READONLY) )) break;
			
			if(FAILED( ctex->LockRect(i,&clr,NULL,D3DLOCK_READONLY) ))
			{
				atex->UnlockRect(i);
				break;
			}

			if(FAILED( dtex->LockRect(i,&dlr,NULL,0) ))
			{
				atex->UnlockRect(i);
				ctex->UnlockRect(i);
				break;
			}

			for(int y=0;y<(int)cd.Height;y++)
			{
				DWORD *cline = (DWORD*)(((byte*)clr.pBits) + clr.Pitch*y);
				DWORD *aline = (DWORD*)(((byte*)alr.pBits) + alr.Pitch*y);
				DWORD *dline = (DWORD*)(((byte*)dlr.pBits) + dlr.Pitch*y);
				for(int x=0;x<(int)cd.Width;x++)
				{
					*dline = (*cline & 0x00FFFFFF) | ((*aline)<<24);
					cline++;
					aline++;
					dline++;
				}
			}

			ctex->UnlockRect(i);
			atex->UnlockRect(i);
			dtex->UnlockRect(i);
		}

		all_ok = true;

	} while(false);

	if(all_ok)
	{
		tex->Release();
		tex = dtex;
	}
	else
	{
		if(dtex) dtex->Release();
	}
	
	atex->Release();

	return true;
}

bool DevTexture::LoadCube(const char *path)
{
	Unload();
	HRESULT res = D3DXCreateCubeTextureFromFile(Dev.GetDevice(),path,(IDirect3DCubeTexture9**)&tex);
	if(!FAILED(res) && tex)
		return true;

	string path1 = FilePathGetPart(path,true,true,false);
	string path2 = FilePathGetPart(path,false,false,true);
	DevTexture t2;

	static const char *SIDES[6] = {
		"_xp", "_xm", "_yp", "_ym", "_zp", "_zm"
	};

	int size = 0;
	for(int i=0;i<6;i++)
	{
		bool ok = false;
		do {
		
			if(!t2.Load((path1 + SIDES[i] + path2).c_str()) || !t2.tex)
				break;

			D3DSURFACE_DESC desc;
			if(FAILED(((IDirect3DTexture9*)t2.tex)->GetLevelDesc(0,&desc)))
				break;

			if(desc.Format!=D3DFMT_A8R8G8B8 && desc.Format!=D3DFMT_X8R8G8B8)
				break;

			if(!size)
			{
				size = desc.Width;
				if(FAILED(Dev->CreateCubeTexture(size,0,D3DUSAGE_AUTOGENMIPMAP,D3DFMT_A8R8G8B8,D3DPOOL_MANAGED,
												 (IDirect3DCubeTexture9**)&tex,NULL)) || !tex)
					 break;
			}

			if(desc.Width!=size || desc.Height!=size)
				break;

			D3DLOCKED_RECT tlr, clr;
			if(FAILED(((IDirect3DTexture9*)t2.tex)->LockRect(0,&tlr,NULL,0)))
				break;

			if(FAILED(((IDirect3DCubeTexture9*)tex)->LockRect((D3DCUBEMAP_FACES)i,0,&clr,NULL,0)))
				break;

			for(int y=0;y<size;y++)
			{
				DWORD *tline = (DWORD*)(((byte*)tlr.pBits) + tlr.Pitch*y);
				DWORD *cline = (DWORD*)(((byte*)clr.pBits) + clr.Pitch*y);
				memcpy(cline,tline,4*size);
			}

			ok = true;
		} while(0);

		if(t2.tex) ((IDirect3DTexture9*)t2.tex)->UnlockRect(0);
		if(tex) ((IDirect3DCubeTexture9*)tex)->UnlockRect((D3DCUBEMAP_FACES)i,0);

		if(!ok)
		{
			Unload();
			return false;
		}
	}

	return true;
}

bool DevTexture::LoadRaw2D(int format,int w,int h,const void *data,bool use_mipmaps)
{
	IDirect3DTexture9 *ttex = NULL;
	
	Unload();

	int bpp = Device::GetSurfaceSize((D3DFORMAT)format,1,1);
	if(bpp<=0)
	{
		assert(!"LoadRaw2D: unsupported format specified");
		return false;
	}

	if(FAILED(Dev->CreateTexture(w,h,use_mipmaps?1:0,use_mipmaps?D3DUSAGE_AUTOGENMIPMAP:0,
										(D3DFORMAT)format,D3DPOOL_MANAGED,&ttex,NULL)))
		return false;

	if(!ttex)
		return false;

	D3DLOCKED_RECT lr;

	bool done = false;
	do
	{
		if(FAILED(ttex->LockRect(0,&lr,NULL,0)))
			break;

		for(int y=0;y<h;y++)
		{
			byte *dst = ((byte*)lr.pBits)+lr.Pitch*y;
			memcpy(dst,((byte*)data)+y*w*bpp,bpp*w);
		}

		ttex->UnlockRect(0);
		done = true;
	} while(0);

	if(!done)
	{
		ttex->Release();
		return false;
	}

	ttex->GenerateMipSubLevels();
	tex = ttex;

	return true;
}

bool DevTexture::CreateEmpty2D(int format,int w,int h,bool create_mipmaps,bool autogen_mipmaps)
{
	IDirect3DTexture9 *ttex = NULL;

	Unload();

	if(FAILED(Dev->CreateTexture(w,h,create_mipmaps?0:1,autogen_mipmaps?D3DUSAGE_AUTOGENMIPMAP:0,
									(D3DFORMAT)format,D3DPOOL_MANAGED,&ttex,NULL)))
		return false;

	if(!ttex)
		return false;

	tex = ttex;

	return true;
}

bool DevTexture::BuildLookup2D(int w,int h,DWORD (*fn)(float,float),bool use_mipmaps)
{
	IDirect3DTexture9 *ttex = NULL;

	Unload();

	if(FAILED(Dev->CreateTexture(w,h,use_mipmaps?0:1,use_mipmaps?D3DUSAGE_AUTOGENMIPMAP:0,
										D3DFMT_A8R8G8B8,D3DPOOL_MANAGED,&ttex,NULL)))
		return false;

	if(!ttex)
		return false;

	D3DLOCKED_RECT lr;
	float u,dx,dy;
	dx = 1.f/float(w);
	dy = 1.f/float(h);

	bool done = false;
	do
	{
		if(FAILED(ttex->LockRect(0,&lr,NULL,0)))
			break;

		for(int y=0;y<h;y++)
		{
			DWORD *data = (DWORD*)(((char*)lr.pBits)+lr.Pitch*y);
			float v = (0.5f+y)*dy;
			for(u=dx*0.5f;u<1.f;u+=dx)
				*data++ = fn(u,v);
		}

		ttex->UnlockRect(0);
		done = true;
	} while(0);

	if(!done)
	{
		ttex->Release();
		return false;
	}

	ttex->GenerateMipSubLevels();
	tex = ttex;

	return true;
}

/*
bool DevTexture::BuildLookup3D(int sx,int sy,int sz,DWORD (*fn)(float,float,float),bool use_mapmaps)
{
	Unload();

	// TODO: implement
	return false;
}
*/

bool DevTexture::BuildLookupCube(int size,DWORD (*fn)(float,float,float),bool use_mipmaps)
{
	IDirect3DCubeTexture9 *ctex = NULL;

	Unload();

	if(FAILED(Dev->CreateCubeTexture(size,use_mipmaps?0:1,use_mipmaps?D3DUSAGE_AUTOGENMIPMAP:0,
										D3DFMT_A8R8G8B8,D3DPOOL_MANAGED,&ctex,NULL)))
		return false;

	if(!ctex)
		return false;

	D3DLOCKED_RECT lr;
	int face;
	float u,v,dt;
	float x,y,z;

	dt = 2.f/float(size);

	for(face=0;face<6;face++)
	{
		if(FAILED(ctex->LockRect((D3DCUBEMAP_FACES)face,0,&lr,NULL,0)))
			break;

		DWORD *data = (DWORD *)lr.pBits;

		for(v=-1.f+dt*0.5f;v<1.f;v+=dt)
			for(u=-1.f+dt*0.5f;u<1.f;u+=dt)
			{
				switch(face)
				{
					case 0: x= 1; y=-v; z=-u; break;
					case 1: x=-1; y=-v; z= u; break;
					case 2: x= u; y= 1; z= v; break;
					case 3: x= u; y=-1; z=-v; break;
					case 4: x= u; y=-v; z= 1; break;
					case 5: x=-u; y=-v; z=-1; break;
				}

				*data++ = fn(x,y,z);
			}

		ctex->UnlockRect((D3DCUBEMAP_FACES)face,0);
	}

	if(face<6)
	{
		ctex->Release();
		return false;
	}

	ctex->GenerateMipSubLevels();
	tex = ctex;

	return true;
}

bool DevTexture::GetRawData(int &width,int &height,vector<DWORD> &data)
{
	width = 0;
	height = 0;
	data.clear();
	if(!tex) return false;
	if(tex->GetType() != D3DRTYPE_TEXTURE)
		return false;

	IDirect3DTexture9 *tex2d = (IDirect3DTexture9 *)tex;
	
	D3DSURFACE_DESC desc;
	if(FAILED(tex2d->GetLevelDesc(0,&desc)))
		return false;

	if(desc.Format!=D3DFMT_A8R8G8B8 && desc.Format!=D3DFMT_X8R8G8B8)
		return false;

	D3DLOCKED_RECT lr;
	if(FAILED(tex2d->LockRect(0,&lr,NULL,0)))
		return false;

	data.resize(desc.Width*desc.Height);
	for(int y=0;y<(int)desc.Height;y++)
	{
		DWORD *src = (DWORD*)(((char*)lr.pBits)+lr.Pitch*y);
		memcpy(&data[y*desc.Width],src,desc.Width*4);
	}

	tex2d->UnlockRect(0);

	width = desc.Width;
	height = desc.Height;

	return true;
}

vec2 DevTexture::GetSize2D()
{
	if(!tex) return vec2(0,0);
	if(tex->GetType() != D3DRTYPE_TEXTURE)
		return vec2(0,0);

	IDirect3DTexture9 *tex2d = (IDirect3DTexture9 *)tex;
	
	D3DSURFACE_DESC desc;
	if(FAILED(tex2d->GetLevelDesc(0,&desc)))
		return vec2(0,0);

	return vec2(float(desc.Width),float(desc.Height));
}

void DevTexture::OnBeforeReset()
{
	// nothing to do here
}

void DevTexture::OnAfterReset()
{
	// nothing to do here
}

void DevTexture::OnPreload()
{
	if(preload_path.size()>0)
		Load(preload_path.c_str());
}

// ******************************** d_txfont.cpp ********************************

// ---- #include "dxfw.h"
// ---> including dxfw.h
// <--- back to d_txfont.cpp

using namespace std;
using namespace base;




static bool ParseKeyValue(const char *&s,string &key,int &value)
{
	ParseWhitespace(s);
	if(!*s) return false;

	string ps;
	ParseString(s,ps);
	const char *p = ps.c_str();
	const char *x = p;
	while(*x && *x!='=') x++;
	if(*x=='=')
	{
		const char *xx = x+1;
		key.assign(p,x);
		value = ParseInt(xx);
		return true;
	}

	return ParseKeyValue(s,key,value);
}


DevTxFont::DevTxFont(const char *face)
{
	path = face;
}


void DevTxFont::Clear()
{
	chars.clear();
	chars.resize(96);
	memset(&chars[0],0,sizeof(CharInfo)*chars.size());
	ker.clear();
    height = 0;
}

bool DevTxFont::Load(const char *name)
{
	Clear();
	path = name;

	vector<string> file;
	if(!NFS.GetFileLines(format("%s.fnt",name).c_str(),file))
		return false;

	texture.Load(format("%s_0",name).c_str());
	tex_size = texture.GetSize2D();

	for(int i=0;i<int(file.size());i++)
	{
		const char *s = file[i].c_str();
		string cmd, key;
		ParseString(s,cmd);
		int value;

		if(cmd=="char")
		{
			CharInfo ch;
			memset(&ch,0,sizeof(ch));
			int id = -1;

			while(ParseKeyValue(s,key,value))
			{
				if(key=="id"      ) id    = value;
				if(key=="x"       ) ch.tx = value;
				if(key=="y"       ) ch.ty = value;
				if(key=="width"   ) ch.tw = value;
				if(key=="height"  ) ch.th = value;
				if(key=="xoffset" ) ch.ox = value;
				if(key=="yoffset" ) ch.oy = value;
				if(key=="xadvance") ch.dx = value;
			}

			if(id>=32 && id<128)
            {
				chars[id-32] = ch;
                if(ch.oy + ch.th > height)
                    height = ch.oy + ch.th;
            }
		}
		else if(cmd=="kerning")
		{
			KerningInfo k;

			while(ParseKeyValue(s,key,value))
			{
				if(key=="first"   ) k.id  |= value << 8;
				if(key=="second"  ) k.id  |= value;
				if(key=="amount"  ) k.adj  = value;
			}

			ker.push_back(k);
		}
	}
	sort(ker.begin(),ker.end());

	return true;
}

int DevTxFont::ComputeLength(const char *s,const char *e)
{
    if(!*s || s==e) return 0;

    int len = 0, last = 0;
    while(*s && s!=e)
    {
        if(*s>=32 && *s<128)
            len += chars[(last=*s)-32].dx;
        s++;
    }
    if(last)
		len += chars[last-32].ox + chars[last-32].tw - chars[last-32].dx;
    return len;
}

void DevTxFont::DrawText(DevCanvas &canvas,int l,float xp,float yp,int align,const base::vec2 &scale,int color,const char *s,const char *e)
{
	canvas.SelectActive(l,texture);
	vector<DevCanvas::Vertex> *_layer = canvas.GetActiveBuffer();
	if(!_layer) return;
	vector<DevCanvas::Vertex> &layer = *_layer;

	xp/=scale.x;
	yp/=scale.y;
    
    if((align&0xF0)==0x10) xp -= (ComputeLength(s,e)+1)/2;
    if((align&0xF0)==0x20) xp -=  ComputeLength(s,e);
    if((align&0x0F)==0x01) yp -= height/2;
    if((align&0x0F)==0x02) yp -= height;

    while(*s && s!=e)
	{
    	int cid = *s++;
		if(cid<32 || cid>=128)
			continue;

		int p = cid-32;
		float x1 = (xp+chars[p].ox)*scale.x;
		float y1 = (yp+chars[p].oy)*scale.y;
		float x2 = x1+chars[p].tw*scale.x;
		float y2 = y1+chars[p].th*scale.y;
		float u1 = float(chars[p].tx)/tex_size.x;
		float v1 = float(chars[p].ty)/tex_size.y;
		float u2 = float(chars[p].tx+chars[p].tw)/tex_size.x;
		float v2 = float(chars[p].ty+chars[p].th)/tex_size.y;

		layer.resize(layer.size()+4);
		DevCanvas::Vertex *v = &layer[layer.size()-4];
    
		v->pos.x = x1;
		v->pos.y = y1;
		v->z = 0;
		v->color = color;
		v->tc.x = u1;
		v->tc.y = v1;
		v++;

		v->pos.x = x2;
		v->pos.y = y1;
		v->z = 0;
		v->color = color;
		v->tc.x = u2;
		v->tc.y = v1;
		v++;

		v->pos.x = x2;
		v->pos.y = y2;
		v->z = 0;
		v->color = color;
		v->tc.x = u2;
		v->tc.y = v2;
		v++;

		v->pos.x = x1;
		v->pos.y = y2;
		v->z = 0;
		v->color = color;
		v->tc.x = u1;
		v->tc.y = v2;
		//v++;

		xp += chars[p].dx;
	}
}

void DevTxFont::DrawTextF(DevCanvas &canvas,int layer,float xp,float yp,int align,const base::vec2 &scale,int color,const char *fmt,...)
{
	string tmp;
	va_list arg;
	va_start(arg,fmt);
	vsprintf(tmp,fmt,arg);
	va_end(arg);
    DrawText(canvas,layer,xp,yp,align,scale,color,tmp.c_str());
}

void DevTxFont::DrawTextF(DevCanvas &canvas,int layer,float xp,float yp,int align,float scale,int color,const char *fmt,...)
{
	string tmp;
	va_list arg;
	va_start(arg,fmt);
	vsprintf(tmp,fmt,arg);
	va_end(arg);
    DrawText(canvas,layer,xp,yp,align,vec2(scale,scale),color,tmp.c_str());
}

float DevTxFont::DrawTextWrap(DevCanvas &canvas,int layer,float xp,float yp,float width,const base::vec2 &scale,int color,const char *s,bool nodraw)
{
	while(*s)
	{
		const char *b = s, *w = s;

		int plen = 0;
		while(*s && *s!='\n' && *s!='\\')
		{
            if(*s<=32) w = s;
			if(*s>=32 && *s<128)
			{
				int len = plen + chars[*s-32].ox + chars[*s-32].tw;
                if(len>width) { s=w; break; }
				plen += chars[*s-32].dx;
			}
			s++;
		}
		if(s==b) s++;
		const char *e = s;

		while(*s==' ') s++;
		if(*s=='\n' || *s=='\\') s++;

		if(!nodraw) DrawText(canvas,layer,xp,yp,0x00,scale,color,b,e);
		yp += height;
	}
    return yp;
}

