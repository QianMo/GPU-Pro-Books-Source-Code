
#include "../framework/dxfw.h"


using namespace std;
using namespace base;

DevEffect fx("render.fx");
DevMesh mesh;

DevTexture tex;
DevFont font("Verdana",13,0,false,false);

vector<string> files;
int selected = -1;
vec3 base_a(1,1,1), base_b(1,1,1);
string base_dir;

void Select(int id)
{
	if(id<0 || id>=files.size())
		return;

	tex.Load( (base_dir + "/" + files[id]).c_str() );

	base_a = vec3(1,1,1);
	base_b = vec3(1,1,1);
	FILE *fp = fopen( (base_dir + "/" + files[id] + ".txt").c_str(),"rt");
	if(fp)
	{
		fscanf(fp,"%f%f%f%f%f%f\n",&base_a.x,&base_a.y,&base_a.z,&base_b.x,&base_b.y,&base_b.z);
		fclose(fp);
	}

	selected = id;
}

void Render()
{
	float a = Dev.GetElapsedTime()*.03f;
	Dev.debug_cam_pos.x = -cos(a)*2.f - sin(a)*.25f;
	Dev.debug_cam_pos.y = sin(a)*2.f - cos(a)*.25f;
	Dev.debug_cam_ypr.x = -a*180/M_PI;

	D3DXMATRIX VP;
	Dev.BuildCameraViewProjMatrix(&VP,Dev.debug_cam_pos,Dev.debug_cam_ypr,70,-1,.001f,1000);

	fx.StartTechnique("tech");
	fx->SetMatrix("MVP",&VP);
	fx->SetTexture("tex",tex);
	fx.SetFloat("rot",0);
	fx.SetFloat3("base_a",base_a);
	fx.SetFloat3("base_b",base_b);
	while(fx.StartPass())
		mesh.DrawSection(0);

	static int _mb = 0;
	vec2 mp = Dev.GetMousePosV();
	int mb = Dev.GetKeyState(VK_LBUTTON) ? 1 : 0;
	int mck = mb & ~_mb;
	_mb = mb;

	for(int i=0;i<files.size();i++)
	{
		Dev.PrintF(&font,5,5+14*i,(i==selected ? 0xFFFFFF80 : 0xFF808080),"%s",files[i].c_str());

		if( mck && mp.x<200 && mp.y>=5+14*i && mp.y<5+14*(i+1) )
			Select(i);
	}
}

int WINAPI WinMain(HINSTANCE,HINSTANCE,LPSTR lpCmdLine,int)
{
    Dev.SetResolution(800,640,false);
    Dev.Init();
    Dev.MainLoop();

	mesh.LoadSphere(1,80,40);
	Dev.debug_cam_pos.x = -3;

	const char *cmdline = lpCmdLine;
	base_dir = ParseString(cmdline);
	if(base_dir.size()<=0)
		base_dir = ".";

	NFS.GetFileList( (base_dir + "/*.txt").c_str(), files );

	for(int i=0;i<files.size();i++)
		files[i] = FilePathGetPart(files[i].c_str(),false,true,false);
	sort(files.begin(),files.end());
	Select(0);
	

    while(Dev.MainLoop())
    {
        Dev->Clear(0,0,D3DCLEAR_TARGET | D3DCLEAR_STENCIL | D3DCLEAR_ZBUFFER,0x000000,1.f,0);
        
		Render();
    }

    return 0;
}
