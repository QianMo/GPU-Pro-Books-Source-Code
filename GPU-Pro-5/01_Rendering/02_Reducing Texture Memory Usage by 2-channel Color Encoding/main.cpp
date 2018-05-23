
#include <allegro.h>
#undef main

#include <winalleg.h>
#include "framework/base.h"

using namespace std;
using namespace base;

#pragma comment(lib,"alleg.lib")


// Weights of source channels (bigger value = allow less error during fitting).
//	Weights are relative to each other.
int weight_R = 1;
int weight_G = 1;
int weight_B = 1;



// Compute fitting color plane and return its normal.
vec3 estimate_image(BITMAP *src)
{

	// Accumulators.
	__int64 _rr = 0;
    __int64 _gg = 0;
    __int64 _bb = 0;
    __int64 _rg = 0;
    __int64 _rb = 0;
    __int64 _gb = 0;

	// Process pixels.
    for(int y=0;y<src->h;y++)
    {
        DWORD *p = (DWORD*)(src->line[y]);
        DWORD *e = p + src->w;
        while(p<e)
        {
			// Extract RGB.
            int c = *p++;
            int r = (c>>16)&0xFF;
            int g = (c>> 8)&0xFF;
            int b = (c    )&0xFF;

			// Apply weights and gamma 2.0.
            r *= r*weight_R;
            g *= g*weight_G;
            b *= b*weight_B;

			// Accumulate.
            _rr += __int64(r)*r;
            _gg += __int64(g)*g;
            _bb += __int64(b)*b;
            _rg += __int64(r)*g;
            _rb += __int64(r)*b;
            _gb += __int64(g)*b;
        }
    }

	// Compute averagesfrom the accumulators.
    int pix = src->w*src->h;
    float rr = float(_rr)/pix;
    float gg = float(_gg)/pix;
    float bb = float(_bb)/pix;
    float rg = float(_rg)/pix;
    float rb = float(_rb)/pix;
    float gb = float(_gb)/pix;

	// Brute force normal fitting.
    vec3 best_n(0,0,0);
    float best_E = 1e18;
    for(int z=-100;z<=100;z++)
        for(int y=-100;y<=100;y++)
            for(int x=-100;x<=100;x++)
            {
                if( !(x|y|z) ) continue;
                vec3 n(x,y,z);
                n.normalize();

				// Compute total sum of squared error values.
                float E =
                    n.x*n.x*rr +
                    n.y*n.y*gg +
                    n.z*n.z*bb +
                    2*n.x*n.y*rg +
                    2*n.x*n.z*rb +
                    2*n.y*n.z*gb;

                if(E<best_E)
                {
					// Apply weights to normal to negate their influence.
                    n.x *= weight_R;
                    n.y *= weight_G;
                    n.z *= weight_B;
                    n.normalize();

					// Update best normal.
                    best_E = E;
                    best_n = n;
                }
            }

    return best_n;
}

// Segment-plane intersection.
//	p1, p2	- segment endpoints (p1-inclusive, p2-exclusive)
//	n		- plane normal (plane always crosses origin)
//
// Returns true if intersection is found and returned via "out".
bool cross_test(const vec3 &p1,const vec3 &p2,const vec3 &n,vec3 &out)
{
    float d1 = p1.dot(n);
    float d2 = p2.dot(n);
    if( (d1<=0 && d2>0) ||
        (d1>=0 && d2<0) )
    {
        float p = -d1/(d2-d1);
        out = p1 + (p2-p1)*p;

        float m = max(out.x,out.y);
        m = max(m,out.z);
        out *= 1.f/m;

        return true;
    }

    return false;
}

// Computes two points where plane with normal "n" crosses unit cube silhouette.
// Returned as "ca" and "cb".
void find_components(const vec3 &n,vec3 &ca,vec3 &cb)
{
	// PTS - unit cube silhouette, as seen from origin.
    static const vec3 PTS[] = {
        vec3(1,0,0),
        vec3(1,1,0),
        vec3(0,1,0),
        vec3(0,1,1),
        vec3(0,0,1),
        vec3(1,0,1)
    };

	// Initialize to safe base colors (white and green) in case we get degenerate normal.
    ca = vec3(1,1,1);
    cb = vec3(0,1,0);

	// Search for intersections.
    vec3 tmp;
    for(int i=0;i<6;i++)
        if(cross_test(PTS[i],PTS[(i+1)%6],n,tmp))
            cb=ca, ca=tmp;
}

// Encode RGB source image into two channels of a new bitmap.
//	n - fitting plane normal vector.
BITMAP *encode_image(BITMAP *src,vec3 n)
{
	// Normalize the normal and find base colors.
    vec3 base_a, base_b;
    n.normalize();
    find_components(n,base_a,base_b);

	// Recompute normal in case it was degenerate.
    n = base_a.cross(base_b).get_normalized();

	vec3 to_scale(0.2126f,0.7152f,0.0722f);

	// Find 2D coordinate frame on the plane.
	//	frame_x is aligned with base_a.
	//	frame_y is perpendicular to frame_x and the normal.
    vec3 frame_x = base_a.get_normalized();
    vec3 frame_y = frame_x.cross(n).get_normalized();

	// fa, fb - base colors as 2D coordinates in plane coordinate frame.
    vec2 fa( frame_x.dot(base_a), frame_y.dot(base_a) );
    vec2 fb( frame_x.dot(base_b), frame_y.dot(base_b) );

	// Create output bitmap.
    BITMAP *dst = create_bitmap(src->w,src->h);

	// Process.
    for(int y=0;y<src->h;y++)
    {
        DWORD *s = (DWORD*)(src->line[y]);
        DWORD *d = (DWORD*)(dst->line[y]);
        DWORD *e = s + src->w;
        while(s<e)
        {
			// Get float RGB values.
            vec3 c;
            int ci = *s++;
            c.x = (ci>>16)&0xFF;
            c.y = (ci>> 8)&0xFF;
            c.z = (ci    )&0xFF;
            c *= 1.f/255;

			// Apply gamma 2.0.
            c.scale_xyz(c);

			// Compute color XY coordinates in 2D frame.
            vec2 cf;
            cf.x = frame_x.dot(c);
            cf.y = frame_y.dot(c);

			// "cf" was direction vector; rotate it by 90 deg to make it 2D line normal.
            cf = cf.get_rotated90();

			// Compute relative distances of base colors to the line.
            float da = fa.dot(cf);
            float db = fb.dot(cf);

			// Compute hue (blend factor between base colors).
            float hue = -da/(db-da+0.00000001f);
            if(hue<0) hue = 0;
            if(hue>1) hue = 1;

            // Compute real luminance and move to gamma 2.0 space.
            float lum = c.dot(to_scale);
            lum = sqrt(lum);

			// Encode data into 2 channels.
			//	R - hue
			//	G - luminace
            int h = int(floor(hue*255+.5f));
            int l = int(floor(lum*255+.5f));
            if(h<0) h = 0;
            if(h>255) h = 255;
            if(l<0) l = 0;
            if(l>255) l = 255;
            *d++ = (h<<16) | (l<<8);
        }
    }

    return dst;
}

// Decode hue/luminance 2-channel image into full RGB.
BITMAP *decode_image(BITMAP *src,const vec3 &base_a,const vec3 &base_b)
{
    BITMAP *dst = create_bitmap(src->w,src->h);

    for(int y=0;y<src->h;y++)
    {
        DWORD *s = (DWORD*)(src->line[y]);
        DWORD *d = (DWORD*)(dst->line[y]);
        DWORD *e = s + src->w;
        while(s<e)
        {
            int ci = *s++;
            float hue = ( (ci>>16)&0xFF )/255.f;
            float lum = ( (ci>> 8)&0xFF )/255.f;

            // Decode linear luminance - apply gamma 2.0.
            lum *= lum;

            // Decode hue color.
            vec3 c = base_a + (base_b-base_a)*hue;

			// Compute luminance of interpolated hue.
            float clum = 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z;

			// Adjust luminance to match desired value.
            c *= lum/(clum + 0.00000001f);

			// Return to original sRGB space.
			//	All sRGB-linear conversions use approximage gamma 2.0,
			//	so all "linear" operations were carried out in not exactly linear space.
			//	Using exact simplifications to gamma 2.0 everywhere make sure that we
			//	return to correct original sRGB space.
            c.x = sqrt(c.x);
            c.y = sqrt(c.y);
            c.z = sqrt(c.z);
            
            // Output color.
            c *= 255;
            int r = int(floor(c.x+.5f));
            int g = int(floor(c.y+.5f));
            int b = int(floor(c.z+.5f));
            if(r<0) r = 0;
            if(r>255) r = 255;
            if(g<0) g = 0;
            if(g>255) g = 255;
            if(b<0) b = 0;
            if(b>255) b = 255;

            *d++ = (r<<16) | (g<<8) | b;
        }
    }

    return dst;
}

// Helper function for Hue-RGB conversion.
float xcos(float f)
{
    f = fmod(f+4.5f,3)-1.5f;
    if(f<-1) return 0;
    if(f>1) return 0;
    f = 2*fabs(f);
    if(f<1) return 1;
    f = 2-f;
    return (3-2*f)*f*f;
}

// Add 64x64 color probe to the bitmap.
void stamp_color_probe(BITMAP *bmp)
{
	if(bmp->w<64 || bmp->h<64)
		return;

    for(int y=0;y<64;y++)
        for(int x=0;x<64;x++)
        {
            vec2 p(x-31.5f,y-31.5f);
            p *= 1.f/31;
            float a = (1-p.length())*27;
            float blk = 1-(1-p.length())*15;
            blk *= 2;
            if(blk<0) blk = 0;
            if(blk>1) blk = 1;
            if(a>1) a = 1;
            if(a<=0) continue;
            a = 1-a;
            blk = 1-blk;

            float hue = atan2(p.x,-p.y)*(1.f/2/M_PI)*3;
            float s = (1-p.length());
            if(s<0) s = 0;
            p.normalize();
            
            float r = xcos(hue  );
            float g = xcos(hue-1);
            float b = xcos(hue-2);
            float m = max(r,g);
            m = max(m,b)/255.f;
            r /= m;
            g /= m;
            b /= m;
            r += (255-r)*s;
            g += (255-g)*s;
            b += (255-b)*s;
            r *= blk;
            g *= blk;
            b *= blk;

            int c = getpixel(bmp,x+bmp->w-64,y+bmp->h-64);
            float cr = (c>>16)&0xFF;
            float cg = (c>> 8)&0xFF;
            float cb = (c    )&0xFF;
            r += (cr-r)*a;
            g += (cr-g)*a;
            b += (cr-b)*a;

            putpixel(bmp,x+bmp->w-64,y+bmp->h-64,(int(r)<<16)|(int(g)<<8)|int(b));
        }
}

void print_usage(char *argv[])
{
	printf("Usage:\n");
	printf("\t%s -encode <input>.bmp [options] <output>.bmp <out_params>.txt\n",argv[0]);
	printf("\t%s -decode <input>.bmp <in_params>.txt <output>.bmp\n",argv[0]);
	printf("\t%s -colorprobe <input>.bmp <output>.bmp\n",argv[0]);
	printf("\t%s -fulldemo <input>.bmp <original_and_transcoded_comparison>.bmp\n",argv[0]);
	printf("\nOptions for encoder:\n");
	printf("\t-weights <wR> <wG> <wB>\t\tSpecify channel importance\n\t\t\t\t\t  (relative integer values 1..1000)\n\t\t\t\t\t  defaults to 1/1/1\n");
	printf("\t-colorprobe\t\t\tStamp color probe in image corner\n");
}

// Main function.
int main(int argc,char *argv[])
{
	int nfiles = 0;
	string files[3];
	bool do_encode = false;
	bool do_decode = false;
	bool do_stamp = false;
	bool full_demo = false;
	int read_weight = -1;

	allegro_init();
	set_color_depth(32);
	set_gfx_mode(GFX_AUTODETECT_WINDOWED,128,64,0,0);
	set_display_switch_mode(SWITCH_BACKGROUND);

	for(int i=1;i<argc;i++)
	{
		const char *cmd = argv[i];
		if(cmd[0]=='-')
		{
				 if(strcmp(cmd,"-encode"    )==0) do_encode = true;
			else if(strcmp(cmd,"-decode"    )==0) do_decode = true;
			else if(strcmp(cmd,"-fulldemo"  )==0) full_demo = true;
			else if(strcmp(cmd,"-colorprobe")==0) do_stamp = true;
			else if(strcmp(cmd,"-weights"   )==0) read_weight = 0;
			else
			{
				print_usage(argv);
				return 1;
			}
		}
		else
		{
			if(read_weight>=0 && read_weight<3)
			{
				int w = ParseInt(cmd);
				if(w<=0 || w>1000)
				{
					print_usage(argv);
					return 1;
				}
				if(read_weight==0) weight_R = w;
				if(read_weight==1) weight_G = w;
				if(read_weight==2) weight_B = w;
				read_weight++;
			}
			else
			{
				if(nfiles<3)
					files[nfiles] = cmd;
				nfiles++;
			}
		}
	}

	if( (read_weight>=0 && read_weight<3) ||
		(!do_encode && read_weight!=-1) )
	{
		print_usage(argv);
		return 1;
	}

	if( do_stamp && !do_encode && !do_decode && !full_demo && nfiles==2 )
	{
		// color probe
		BITMAP *src = load_bitmap(files[0].c_str(),NULL);
		if(!src) { printf("Can't load '%s' as input texture\n",files[0].c_str()); return 1; }

		stamp_color_probe(src);

		if(save_bitmap(files[1].c_str(),src,NULL)) { printf("Can't write image with color probe to '%s'\n",files[1].c_str()); return 1; }

		printf("Color probe added successfully.\n");
		return 0;
	}

	if( full_demo && !do_encode && !do_decode && !do_stamp && nfiles==2 )
	{
		// encode
		BITMAP *src = load_bitmap(files[0].c_str(),NULL);
		if(!src) { printf("Can't load '%s' as input texture\n",files[0].c_str()); return 1; }

		vec3 est = estimate_image(src);
		stamp_color_probe(src);

		BITMAP *encoded = encode_image(src,est);

		vec3 base_a, base_b;
		find_components(est,base_a,base_b);

		// decode
		BITMAP *decoded = decode_image(encoded,base_a,base_b);
		BITMAP *final = create_bitmap(src->w + decoded->w,src->h);

		blit(src,final,0,0,0,0,src->w,src->h);
		blit(decoded,final,0,0,src->w,0,decoded->w,decoded->h);

		if(save_bitmap(files[1].c_str(),final,NULL)) { printf("Can't write final image to '%s'\n",files[1].c_str()); return 1; }

		printf("Demo comparison image created successfully.\n");
		return 0;
	}

	if( (!do_encode && !do_decode) ||
		(do_encode && do_decode) ||
		full_demo ||
		(do_decode && do_stamp) ||
		(nfiles!=3) )
	{
		print_usage(argv);
		return 1;
	}

	if( do_encode )
	{
		// encode
		BITMAP *src = load_bitmap(files[0].c_str(),NULL);
		if(!src) { printf("Can't load '%s' as input texture\n",files[0].c_str()); return 1; }

		vec3 est = estimate_image(src);
		if(do_stamp) stamp_color_probe(src);

		BITMAP *encoded = encode_image(src,est);

		if(save_bitmap(files[1].c_str(),encoded,NULL)) { printf("Can't write encoded image to '%s'\n",files[1].c_str()); return 1; }

		vec3 base_a, base_b;
		find_components(est,base_a,base_b);
		FILE *fp = fopen(files[2].c_str(),"wt");
		if(!fp) { printf("Can't write color space parameters to '%s'\n",files[2].c_str()); return 1; }
		fprintf(fp,"%f %f %f %f %f %f\n",base_a.x,base_a.y,base_a.z,base_b.x,base_b.y,base_b.z);
		fclose(fp);

		printf("Encoding successfull.\n");
	}
	else
	{
		// decode
		BITMAP *src = load_bitmap(files[0].c_str(),NULL);
		if(!src) { printf("Can't load '%s' as encoded input texture\n",files[0].c_str()); return 1; }

		vec3 base_a(0,0,0), base_b(0,0,0);
		FILE *fp = fopen(files[1].c_str(),"rt");
		if(!fp) { printf("Can't load color space parameters from '%s'\n",files[1].c_str()); return 1; }
		fscanf(fp,"%f%f%f%f%f%f\n",&base_a.x,&base_a.y,&base_a.z,&base_b.x,&base_b.y,&base_b.z);
		fclose(fp);

		if(base_a.length()<=0 || base_b.length()<=0) { printf("Invalid color space parameters in file '%s'\n",files[1].c_str()); return 1; }

		BITMAP *decoded = decode_image(src,base_a,base_b);
		if(save_bitmap(files[2].c_str(),decoded,NULL)) { printf("Can't write decoded image to '%s'\n",files[2].c_str()); return 1; }
		
		printf("Decoding successfull.\n");
	}

	return 0;
}
