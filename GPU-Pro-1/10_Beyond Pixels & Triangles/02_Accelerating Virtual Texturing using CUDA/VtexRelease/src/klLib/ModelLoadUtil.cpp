/**
 *
 *  This software module was originally developed for research purposes,
 *  by Multimedia Lab at Ghent University (Belgium).
 *  Its performance may not be optimized for specific applications.
 *
 *  Those intending to use this software module in hardware or software products
 *  are advized that its use may infringe existing patents. The developers of 
 *  this software module, their companies, Ghent Universtity, nor Multimedia Lab 
 *  have any liability for use of this software module or modifications thereof.
 *
 *  Ghent University and Multimedia Lab (Belgium) retain full right to modify and
 *  use the code for their own purpose, assign or donate the code to a third
 *  party, and to inhibit third parties from using the code for their products. 
 *
 *  This copyright notice must be included in all copies or derivative works.
 *
 *  For information on its use, applications and associated permission for use,
 *  please contact Prof. Rik Van de Walle (rik.vandewalle@ugent.be). 
 *
 *  Detailed information on the activities of
 *  Ghent University Multimedia Lab can be found at
 *  http://multimedialab.elis.ugent.be/.
 *
 *  Copyright (c) Ghent University 2004-2009.
 *
 **/

#include "shared.h"
#include "ModelLoadUtil.h"

/////////////////////////// Tangent space calculation /////////////////////////////////


static void TangentForTri(const float *v0, const float *v1, const float *v2,
				   const float *st0, const float*st1, const float *st2,
				   float *Tangent, float *Binormal) 
{
	klVec3 vec1, vec2;
	klVec3 planes[3];
	int i;

	for (i=0; i<3; i++) {
		vec1[0] = v1[i]-v0[i];
		vec1[1] = st1[0]-st0[0];
		vec1[2] = st1[1]-st0[1];
		vec2[0] = v2[i]-v0[i];
		vec2[1] = st2[0]-st0[0];
		vec2[2] = st2[1]-st0[1];
		vec1.normalize();
		vec2.normalize();
		planes[i].cross(vec1,vec2);
	}

	Tangent[0] = -planes[0][1]/planes[0][0];
	Tangent[1] = -planes[1][1]/planes[1][0];
	Tangent[2] = -planes[2][1]/planes[2][0];
	Binormal[0] = -planes[0][2]/planes[0][0];
	Binormal[1] = -planes[1][2]/planes[1][0];
	Binormal[2] = -planes[2][2]/planes[2][0];
	//VectorNormalize(Tangent); //is this needed?
	//VectorNormalize(Binormal);
}

static void ClosestPointOnLine(const klVec3 &a,const klVec3 &b,const klVec3 &p, klVec3 &res) {
	klVec3 c,V;
	float d,t ;

	// a-b is the line, p the point in question
	c = p-a;
	V = b-a;
	d = V.length();
	V.normalize();
	t = V*c;

	// Check to see if t is beyond the extents of the line segment
	if (t < 0.0f)
	{
		res = a;
	}
	if (t > d)
	{
		res = b;
	}
	// Return the point between a and b
	V = V*t;
	res = a+V;
}

static void Orthogonalize(const klVec3 &v1,const klVec3 &v2, klVec3 &res) {
	klVec3 v2ProjV1;
	klVec3 iV1;

	iV1 = v1 * -1.0f;
	ClosestPointOnLine(v1, iV1, v2, v2ProjV1);
	res = v2 - v2ProjV1;
	res.normalize();
}

void CalculateTangentSpace(std::vector<klVertex> &verts, std::vector<unsigned int> &indices) {

    // To strore the amount of normals per vertex
    int	*numNormals = (int *)malloc(verts.size() * sizeof(int)); 
	if (!numNormals) {
		klFatalError("Failed to allocate memory\n");
	}

    //set temp to zero
    for (size_t i=0; i<verts.size(); i++) {
        verts[i].tangent.zero();
		verts[i].binormal.zero();
	}

	//for all tris
	for (size_t i=0; i<indices.size()/3; i++) {
		klVec3 tangent;
		klVec3 binormal;
		TangentForTri(verts[indices[i*3+0]].xyz.toPtr(),
                      verts[indices[i*3+1]].xyz.toPtr(),
                      verts[indices[i*3+2]].xyz.toPtr(),
                      verts[indices[i*3+0]].uv.toPtr(),
                      verts[indices[i*3+1]].uv.toPtr(),
                      verts[indices[i*3+2]].uv.toPtr(),
                      tangent.toPtr(), binormal.toPtr());
		//for all vertices in the tri
		for (size_t j=0; j<3; j++) {
			int l = indices[i*3+j];
			verts[l].tangent += tangent;
			verts[l].binormal += binormal;
			numNormals[l]++;
		}
	}

	//calculate average
	for (size_t i=0; i<verts.size(); i++) {

		if (!numNormals[i]) continue;

		verts[i].tangent.normalize();
		verts[i].binormal.normalize();

        klVec3 normal = verts[i].normal;
        Orthogonalize(normal, verts[i].tangent, verts[i].tangent);
		Orthogonalize(normal, verts[i].binormal, verts[i].binormal);
	}

	free(numNormals);
}

/////////////////////////// Make render models /////////////////////////////////

static klMaterial *GetMaterial(const char *name) {
    return materialManager.getForName(name);
}

void ProcessDiscModel( DiscSurfaceList &dsl, klModel * dstModel ) {
    std::vector<klVertex> finalVerts;
    std::vector<unsigned int> finalIndices;
    char lastMaterial[64];

    if ( !dsl.size() ) {
        klFatalError("Empty model");
    }

    // Sort the surfaces by material
    for (size_t i=0; i<dsl.size()-1; i++) {
        for (size_t j=0; j<dsl.size()-1-i; j++) {
            if (dsl[j+1].material < dsl[j].material) {
                DiscSurface tmp = dsl[j];
                dsl[j] = dsl[j+1];
                dsl[j+1] = tmp;
            }
        }
    }

    // Run over all the surfaces, merge with the same material
    // and create vertex and index lists...
    for (size_t surf=0; surf<dsl.size(); surf++ ) {
        DiscSurface &ds = dsl[surf];

        assert(ds.xyz != NULL);
        assert(ds.xyzFaces->size() == ds.uvFaces->size());
        assert(ds.xyzFaces->size() == ds.normalFaces->size());

        if ( strcmp(lastMaterial,ds.material) != 0 ) {

            if ( finalIndices.size() ) {
                klSurface surf;
                //CalculateTangentSpace(finalVerts,finalIndices);
                surf.vertices = new klVertexBuffer(&finalVerts[0],(int)finalVerts.size());
                surf.indices = new klIndexBuffer(&finalIndices[0],(int)finalIndices.size());
                surf.material = GetMaterial(lastMaterial);
                dstModel->addSurface(surf);
            }

            finalVerts.clear();
            finalIndices.clear();
        }
        strcpy(lastMaterial,ds.material);

        for ( size_t face=0; face<ds.xyzFaces->size(); face+=3 ) {
            for ( int corner=0; corner<3; corner++ ) {
                // Create a klVertex struct
                klVertex temp;
                temp.xyz = (*ds.xyz)[(*ds.xyzFaces)[face+corner]];
                if ( ds.uv ) {
                    temp.uv = (*ds.uv)[(*ds.uvFaces)[face+corner]];
                } else {
                    temp.uv.zero();
                }
                if ( ds.normal ) {
                    temp.normal = (*ds.normal)[(*ds.normalFaces)[face+corner]];
                } else {
                    temp.normal.zero();
                }
                if ( ds.tangent ) {
                    temp.tangent = (*ds.tangent)[(*ds.tangentFaces)[face+corner]];
                } else {
                    temp.tangent.zero();
                }
                if ( ds.binormal ) {
                    temp.binormal = (*ds.binormal)[(*ds.binormalFaces)[face+corner]];
                } else {
                    temp.binormal.zero();
                }
                if ( ds.color ) {
                    klVec3 col = (*ds.color)[(*ds.colorFaces)[face+corner]];
                    unsigned char r = (unsigned char)(col[0]*255.0f);
                    unsigned char g = (unsigned char)(col[1]*255.0f);
                    unsigned char b = (unsigned char)(col[2]*255.0f);
                    temp.color = (r<<16) | (g<<8) | b;
                } else {
                    temp.color = 0xFFFFFFFF;
                }

                // Check if an identical vert aready exists
                unsigned int vert;
                size_t vertSize = finalVerts.size();
                klVertex *finalVertsPtr = (vertSize) ? &finalVerts[0] : NULL;
			    for (vert=0; vert<vertSize; vert++) {
				    if ( memcmp(&temp, finalVertsPtr+vert, sizeof(klVertex)) == 0 ) {
					    break;
				    }
			    }

                // If a new one needs to be allocated add it to the list
			    if (vert == finalVerts.size()) {
				    finalVerts.push_back(temp);
			    }

                // And store the index in the face list
                finalIndices.push_back(vert);
            }
        }
    }

    if ( finalIndices.size() ) {
        klSurface surf;
        //CalculateTangentSpace(finalVerts,finalIndices);
        surf.vertices = new klVertexBuffer(&finalVerts[0],(int)finalVerts.size());
        surf.indices = new klIndexBuffer(&finalIndices[0],(int)finalIndices.size());
        surf.material = GetMaterial(lastMaterial);
        dstModel->addSurface(surf);
    }
}
