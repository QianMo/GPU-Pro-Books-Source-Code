// ---------------- Library -- generated on 4.9.2013  0:33 ----------------



// ******************************** dxfw.h ********************************

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

