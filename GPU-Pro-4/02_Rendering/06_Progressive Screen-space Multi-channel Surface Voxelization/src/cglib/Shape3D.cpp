//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//  Computer Graphics Library                                               //
//  Athanasios Gaitatzes (gaitat at yahoo dot com), 2009                    //
//                                                                          //
//  For manuals, help and instructions, please visit:                       //
//  http://graphics.cs.aueb.gr/graphics/                                    //
//                                                                          //
//  http://astronomy.swin.edu.au/~pbourke/geometry/supershape3d/            //
//  http://astronomy.swin.edu.au/~pbourke/geometry                          //
//////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <string.h>

#include "Shape3D.h"

#ifdef WIN32
#pragma warning (disable : 4996)
#endif

Shape3D::Shape3D(_shapeType type, float radius, int depth)
{
    if (type == SHAPE_NONE)
    {
        EAZD_TRACE ("Shape3D::Shape3D() : ERROR - No type specified.");
        return;
    }

    strcpy(filename,"_Shape");

    vertices = NULL;
    normals = NULL;
    texcoords[0] = NULL;
    texcoords[1] = NULL;
    texcoords[2] = NULL;
    materials = NULL;
    groups = NULL;
    spatialSubd = NULL;
    center = Vector3D(0,0,0);

    numfaces = numvertices = numnormals =
        numtexcoords[0] = numtexcoords[1] = numtexcoords[2] =
        numgroups = nummaterials = 0;

    // when N1 and N2 are != 1.0 then we have the super-shapes

         if (type == SHAPE_PLANE) ;
    else if (type == SHAPE_SPHERE)
        sphere (radius, depth);
    else if (type == SHAPE_CYLINDER) ;

    else if (type == SHAPE_ELLIPSOID)
        ellipsoid (1.0f, 1.0f, radius, depth);
    else if (type == SHAPE_HYPERBOLOID_OF_ONE_SHEET)
        hyperboloid_one_sheet (1.0f, 1.0f, radius, depth);

    else if (type == SHAPE_SUPER_ELLIPSOID)
                // N1,   N2
        ellipsoid (0.2f, 1.f, radius, depth);
    else if (type == SHAPE_SUPER_HYPERBOLOID_OF_ONE_SHEET)
        hyperboloid_one_sheet (3.f, 2.f, radius, depth);

    else if (type == SHAPE_TORUS)
            // R0,  R1,  N1,  N2
        torus (4.0f, 1.5f, 1.0f, 1.0f, radius, depth);
    else if (type == SHAPE_SUPER_TOROID)
            // R0,  R1,  N1,  N2
        torus (3.0f, 1.5f, 1.5f, 0.5f, radius, depth);

    else if (type == SHAPE_SUPER_SHAPE)
                 // M,   A,   B,   N1,     N2,    N3
        superShape (7.0f, 1.0f, 1.0f, 20.45f, -0.33f, -3.54f,
                    6.0f, 1.0f, 1.0f, -0.96f,  4.46f,  0.52f,
                    radius, depth);

    processData();
}

inline float
powerf (float val, float p)
{
    if (fabsf (val) < 0.000001f)
        return 0.f;

    int sign = (val < 0 ? -1 : 1);
    return sign * powf (fabsf (val), p);
}

void Shape3D::sphere (float radius, int depth)
{
    int     i, j, idx;               /* i/phi    j/theta */
    float   phi, theta, sin_phi, cos_phi, sin_theta, cos_theta;

    float   step       = (float) (PI / powf (2.f, (float) depth));
    float   fphiSize   = (PI / step) + 1.f;
    float   fthetaSize = (2.f * PI / step) + 1.f;
    int     phiSize    = (int) fphiSize;
    int     thetaSize  = (int) fthetaSize;
    float   phiMin     = -(PI / 2.f);
    float   thetaMin   = -PI;

    // contains the phi indices where we should create triangles
    // instead of quads. usually the noth and south poles
    int phiTrisNorth;
    int phiTrisSouth;

    Vector3D * coords = (Vector3D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector3D));
    Vector3D * norms  = (Vector3D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector3D));
    Vector2D * textr  = (Vector2D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector2D));

    for (i = 0; i < phiSize; i++)
    for (j = 0; j < thetaSize; j++)
    {
        phi   = phiMin   + i * step;
        theta = thetaMin + j * step;

        idx = i * thetaSize + j;

        sin_phi   = sinf (phi);   cos_phi   = cosf (phi);
        sin_theta = sinf (theta); cos_theta = cosf (theta);

        coords[idx].x = cos_phi * cos_theta;
        coords[idx].y = cos_phi * sin_theta;
        coords[idx].z = sin_phi;

        coords[idx] *= radius;

        norms [idx] = coords[idx];
        norms [idx].normalize ();

        textr [idx].x = j / (fthetaSize - 1.0f);
        textr [idx].y = i / (fphiSize - 1.0f);
    }

    numfaces = 2 * (thetaSize - 1) * (phiSize - 2);
    phiTrisNorth = 0;
    phiTrisSouth = phiSize - 2;

    createGeometry (numfaces, thetaSize, phiSize, phiTrisNorth, phiTrisSouth,
                    coords, norms, textr);

    /* release allocated storage */
    free (coords);
    free (norms);
    free (textr);
} // end of sphere

void Shape3D::ellipsoid (float N1, float N2, float radius, int depth)
{
    int     i, j, idx;               /* i/phi    j/theta */
    float   phi, theta, sin_phi, cos_phi, sin_theta, cos_theta;

    float   step       = (float) (PI / powf (2.f, (float) depth));
    float   fphiSize   = (PI / step) + 1.f;
    float   fthetaSize = (2.f * PI / step) + 1.f;
    int     phiSize    = (int) fphiSize;
    int     thetaSize  = (int) fthetaSize;
    float   phiMin     = -(PI / 2.f);
    float   thetaMin   = -PI;

    // contains the phi indices where we should create triangles
    // instead of quads. usually the noth and south poles
    int phiTrisNorth;
    int phiTrisSouth;

    Vector3D * coords = (Vector3D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector3D));
    Vector3D * norms  = (Vector3D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector3D));
    Vector2D * textr  = (Vector2D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector2D));

    for (i = 0; i < phiSize; i++)
    for (j = 0; j < thetaSize; j++)
    {
        phi   = phiMin   + i * step;
        theta = thetaMin + j * step;

        idx = i * thetaSize + j;

        sin_phi   = sinf (phi);   cos_phi   = cosf (phi);
        sin_theta = sinf (theta); cos_theta = cosf (theta);

        coords[idx].x = powerf (cos_phi, N1) * powerf (cos_theta, N2);
        coords[idx].y = powerf (cos_phi, N1) * powerf (sin_theta, N2);
        coords[idx].z = powerf (sin_phi, N1);

        coords[idx] *= radius;

        norms [idx].x = powerf (cos_phi, (2.f - N1)) * powerf (cos_theta, (2.f - N2));
        norms [idx].y = powerf (cos_phi, (2.f - N1)) * powerf (sin_theta, (2.f - N2));
        norms [idx].z = powerf (sin_phi, (2.f - N1));

        norms [idx] *= 1.f / radius;
        norms [idx].normalize ();

        textr [idx].x = j / (fthetaSize - 1.f);
        textr [idx].y = i / (fphiSize - 1.f);
    }

    numfaces = 2 * (thetaSize - 1) * (phiSize - 2);
    phiTrisNorth = 0;
    phiTrisSouth = phiSize - 2;

    createGeometry (numfaces, thetaSize, phiSize, phiTrisNorth, phiTrisSouth,
                    coords, norms, textr);

    /* release allocated storage */
    free (coords);
    free (norms);
    free (textr);
} // end of ellipsoid

void Shape3D::hyperboloid_one_sheet (float N1, float N2, float radius, int depth)
{
    int     i, j, idx;               /* i/phi    j/theta */
    float   phi, theta, sin_phi, cos_phi, sin_theta, cos_theta;
    float               sinh_phi, cosh_phi;

    float   step       = (float) (PI / powf (2.f, (float) depth));
    float   fphiSize   = (PI / step) + 1.f;
    float   fthetaSize = (2.f * PI / step) + 1.f;
    int     phiSize    = (int) fphiSize;
    int     thetaSize  = (int) fthetaSize;
    float   phiMin     = -(PI / 2.f);
    float   thetaMin   = -PI;

    // contains the phi indices where we should create triangles
    // instead of quads. usually the noth and south poles
    int phiTrisNorth;
    int phiTrisSouth;

    Vector3D * coords = (Vector3D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector3D));
    Vector3D * norms  = (Vector3D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector3D));
    Vector2D * textr  = (Vector2D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector2D));

    for (i = 0; i < phiSize; i++)
    for (j = 0; j < thetaSize; j++)
    {
        phi   = phiMin   + i * step;
        theta = thetaMin + j * step;

        idx = i * thetaSize + j;

        sin_phi   = sinf (phi);   cos_phi   = cosf (phi);
        sinh_phi  = sinhf (phi);  cosh_phi  = coshf (phi);
        sin_theta = sinf (theta); cos_theta = cosf (theta);

        coords[idx].x = powerf (cosh_phi, N1) * powerf (cos_theta, N2);
        coords[idx].y = powerf (cosh_phi, N1) * powerf (sin_theta, N2);
        coords[idx].z = powerf (sinh_phi, N1);

        coords[idx] *= radius;

        norms [idx].x = powerf (cosh_phi, (2.f - N1)) * powerf (cos_theta, (2.f - N2));
        norms [idx].y = powerf (cosh_phi, (2.f - N1)) * powerf (sin_theta, (2.f - N2));
        norms [idx].z = powerf (sinh_phi, (2.f - N1));

        norms [idx] *= 1.f / radius;
        norms [idx].normalize ();

        textr [idx].x = j / (fthetaSize - 1.f);
        textr [idx].y = i / (fphiSize - 1.f);
    }

    numfaces = 2 * (thetaSize - 1) * (phiSize - 1);
    phiTrisNorth = -1;
    phiTrisSouth = -1;

    createGeometry (numfaces, thetaSize, phiSize, phiTrisNorth, phiTrisSouth,
                    coords, norms, textr);

    /* release allocated storage */
    free (coords);
    free (norms);
    free (textr);
} // end of hyperboloid_one_sheet

void Shape3D::torus (float R0, float R1, float N1, float N2,
                     float radius, int depth)
{
    int     i, j, idx;               /* i/phi    j/theta */
    float   phi, theta, sin_phi, cos_phi, sin_theta, cos_theta;

    EAZD_ASSERTALWAYS (R0 >= R1);

    float   step       = (float) (PI / powf (2.f, (float) depth));
    float   fphiSize   = (2.f * PI / step) + 1.f;
    float   fthetaSize = (2.f * PI / step) + 1.f;
    int     phiSize    = (int) fphiSize;
    int     thetaSize  = (int) fthetaSize;
    float   phiMin     = -PI;
    float   thetaMin   = -PI;

    // contains the phi indices where we should create triangles
    // instead of quads. usually the noth and south poles
    int phiTrisNorth;
    int phiTrisSouth;

    Vector3D * coords = (Vector3D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector3D));
    Vector3D * norms  = (Vector3D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector3D));
    Vector2D * textr  = (Vector2D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector2D));

    for (i = 0; i < phiSize; i++)
    for (j = 0; j < thetaSize; j++)
    {
        phi   = phiMin   + i * step;
        theta = thetaMin + j * step;

        idx = i * thetaSize + j;

        sin_phi   = sinf (phi);   cos_phi   = cosf (phi);
        sin_theta = sinf (theta); cos_theta = cosf (theta);

        // R0 phi the torus radius - R1 phi the circle radius
        coords[idx].x = (R0 + R1 * powerf (cos_phi, N1)) * powerf (cos_theta, N2);
        coords[idx].y = (R0 + R1 * powerf (cos_phi, N1)) * powerf (sin_theta, N2);
        coords[idx].z =       R1 * powerf (sin_phi, N1);

        coords[idx] *= radius;

        norms [idx].x = powerf (cos_phi, (2.f - N1)) * powerf (cos_theta, (2.f - N2));
        norms [idx].y = powerf (cos_phi, (2.f - N1)) * powerf (sin_theta, (2.f - N2));
        norms [idx].z = powerf (sin_phi, (2.f - N1));

        norms [idx] *= 1.f / radius;
        norms [idx].normalize ();

        textr [idx].x = j / (fthetaSize - 1.0f);
        textr [idx].y = i / (fphiSize - 1.0f);
    }

    numfaces = 2 * (thetaSize - 1) * (phiSize - 1);
    phiTrisNorth = -1;
    phiTrisSouth = -1;

    createGeometry (numfaces, thetaSize, phiSize, phiTrisNorth, phiTrisSouth,
                    coords, norms, textr);

    /* release allocated storage */
    free (coords);
    free (norms);
    free (textr);
} // end of torus

float
super (float theta, float M, float A, float B, float N1, float N2, float N3)
{
    return 1.f / powerf (powerf (fabsf (cosf (theta * M / 4.f) / A), N2) +
                         powerf (fabsf (sinf (theta * M / 4.f) / B), N3), 1.f / N1);
}

void Shape3D::superShape (
            float M1, float A1, float B1, float N11, float N12, float N13,
            float M2, float A2, float B2, float N21, float N22, float N23,
            float radius, int depth)
{
    int     i, j, idx;               /* i/phi    j/theta */
    float   phi, theta, sin_phi, cos_phi, sin_theta, cos_theta;

    float   step       = (float) (PI / powf (2.f, (float) depth));
    float   fphiSize   = (PI / step) + 1.f;
    float   fthetaSize = (2.f * PI / step) + 1.f;
    int     phiSize    = (int) fphiSize;
    int     thetaSize  = (int) fthetaSize;
    float   phiMin     = -(PI / 2.f);
    float   thetaMin   = -PI;

    // contains the phi indices where we should create triangles
    // instead of quads. usually the noth and south poles
    int phiTrisNorth;
    int phiTrisSouth;

    Vector3D * coords = (Vector3D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector3D));
    Vector3D * norms  = (Vector3D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector3D));
    Vector2D * textr  = (Vector2D *) malloc (phiSize * thetaSize *
                                             sizeof (Vector2D));

    for (i = 0; i < phiSize; i++)
    for (j = 0; j < thetaSize; j++)
    {
        phi   = phiMin   + i * step;
        theta = thetaMin + j * step;

        idx = i * thetaSize + j;

        sin_phi   = sinf (phi);   cos_phi   = cosf (phi);
        sin_theta = sinf (theta); cos_theta = cosf (theta);

        coords[idx].x = super (phi,   M2, A2, B2, N21, N22, N23) * cos_phi *
                        super (theta, M1, A1, B1, N11, N12, N13) * cos_theta;
        coords[idx].y = super (phi,   M2, A2, B2, N21, N22, N23) * cos_phi *
                        super (theta, M1, A1, B1, N11, N12, N13) * sin_theta;
        coords[idx].z = super (phi,   M2, A2, B2, N21, N22, N23) * sin_phi;

        coords[idx] *= radius;

        norms [idx] = coords[idx];
        norms [idx].normalize ();

        textr [idx].x = j / (fthetaSize - 1.0f);
        textr [idx].y = i / (fphiSize - 1.0f);
    }

    numfaces = 2 * (thetaSize - 1) * (phiSize - 2);
    phiTrisNorth = 0;
    phiTrisSouth = phiSize - 2;

    createGeometry (numfaces, thetaSize, phiSize, phiTrisNorth, phiTrisSouth,
                    coords, norms, textr);

    /* release allocated storage */
    free (coords);
    free (norms);
    free (textr);
} // end of superShape

void Shape3D::createGeometry (int numfaces, int thetaSize, int phiSize,
    int phiTrisNorth, int phiTrisSouth,
    Vector3D * coords, Vector3D * norms, Vector2D * textr)
{
    int     i, j;               /* i/phi    j/theta */

    numvertices = 3 * numfaces;
    numnormals = numvertices;
    numtangents = numvertices;
    numtexcoords[0] = numvertices;
    numgroups = 1;
    nummaterials = 1;

    // Initialize material
    materials = (Material3D *) malloc (nummaterials * sizeof (Material3D));
    materials[0] = Material3D ();
    materials[0].has_texture[MATERIAL_MAP_DIFFUSE0] = true;
    materials[0].texturemap[MATERIAL_MAP_DIFFUSE0] = loadTexture ("checker.rgb");

    // Initialize group
    groups = (PrimitiveGroup3D *) malloc (numgroups * sizeof (PrimitiveGroup3D));
    groups[0] = PrimitiveGroup3D ();

    groups[0].numfaces = 0;
    groups[0].mtrlIdx = 0;    // this is the id of the material
    groups[0].has_normals = true;
    groups[0].has_texcoords[0] = true;
    strcpy(groups[0].name,"_Default_Group");
#ifdef BUFFER_OBJECT
    groups[0].faces = (Face3D *) malloc (numfaces * sizeof (Face3D));
    groups[0].fc_normals = (Vector3D *) malloc (numfaces * sizeof (Vector3D));
#else // COMPILE_LIST
    groups[0].faces = (Triangle3D *) malloc (numfaces * sizeof (Triangle3D));
#endif

    // Allocating memory
    vertices = (Vector3D *) malloc (numvertices * sizeof (Vector3D));
    if (numnormals != 0)
        normals = (Vector3D *) malloc (numnormals * sizeof (Vector3D));
    if (numtangents != 0)
        tangents = (Vector3D *) malloc (numtangents * sizeof (Vector3D));
    if (numtexcoords[0] != 0)
        texcoords[0] = (Vector2D *) malloc (numtexcoords[0] * sizeof (Vector2D));

    int numverts, numnorms, numtexts, numfcs = 0;
    numverts = numnorms = numtexts = 0;

    /* Print the Conectivity data of the points */
    for (i = 0; i < thetaSize - 1; i++)
    for (j = 0; j < phiSize - 1; j++)
    {
        // the north poles are triangles
        if (phiTrisNorth == j)
        {
#ifdef BUFFER_OBJECT
            Face3D *f = &(groups[0].faces[numfcs]);
#else // COMPILE_LIST
            Triangle3D *f = &(groups[0].faces[numfcs]);
#endif

            f->vertIdx[0] = numverts;
#ifndef BUFFER_OBJECT
            f->normIdx[0] = numnorms;
            f->texcIdx[0] = numtexts;
#endif

            // vertex 0
            vertices    [numverts] = coords[(j + 1) * thetaSize + (i + 1)];
            normals     [numnorms] = norms [(j + 1) * thetaSize + (i + 1)];
            texcoords[0][numtexts] = textr [(j + 1) * thetaSize + (i + 1)];

            numverts++; numnorms++; numtexts++;

            f->vertIdx[1] = numverts;
#ifndef BUFFER_OBJECT
            f->normIdx[1] = numnorms;
            f->texcIdx[1] = numtexts;
#endif

            // vertex 1
            vertices    [numverts] = coords[(j + 1) * thetaSize +  i];
            normals     [numnorms] = norms [(j + 1) * thetaSize +  i];
            texcoords[0][numtexts] = textr [(j + 1) * thetaSize +  i];

            numverts++; numnorms++; numtexts++;

            f->vertIdx[2] = numverts;
#ifndef BUFFER_OBJECT
            f->normIdx[2] = numnorms;
            f->texcIdx[2] = numtexts;
#endif

            // vertex 2
            vertices    [numverts] = coords[ j      * thetaSize +  i];
            normals     [numnorms] = norms [ j      * thetaSize +  i];
            texcoords[0][numtexts] = textr [ j      * thetaSize +  i];

            numverts++; numnorms++; numtexts++;

            numfcs++;
            groups[0].numfaces++;
        }

        // the north poles are triangles
        else
        if (phiTrisSouth == j)
        {
#ifndef BUFFER_OBJECT
            Triangle3D *f = &(groups[0].faces[numfcs]);
#else // COMPILE_LIST
            Face3D *f = &(groups[0].faces[numfcs]);
#endif

            f->vertIdx[0] = numverts;
#ifndef BUFFER_OBJECT
            f->normIdx[0] = numnorms;
            f->texcIdx[0] = numtexts;
#endif

            // vertex 0
            vertices    [numverts] = coords[ j      * thetaSize +  i];
            normals     [numnorms] = norms [ j      * thetaSize +  i];
            texcoords[0][numtexts] = textr [ j      * thetaSize +  i];

            numverts++; numnorms++; numtexts++;

            f->vertIdx[1] = numverts;
#ifndef BUFFER_OBJECT
            f->normIdx[1] = numnorms;
            f->texcIdx[1] = numtexts;
#endif

            // vertex 1
            vertices    [numverts] = coords[ j      * thetaSize + (i + 1)];
            normals     [numnorms] = norms [ j      * thetaSize + (i + 1)];
            texcoords[0][numtexts] = textr [ j      * thetaSize + (i + 1)];

            numverts++; numnorms++; numtexts++;

            f->vertIdx[2] = numverts;
#ifndef BUFFER_OBJECT
            f->normIdx[2] = numnorms;
            f->texcIdx[2] = numtexts;
#endif

            // vertex 2
            vertices    [numverts] = coords[(j + 1) * thetaSize + (i + 1)];
            normals     [numnorms] = norms [(j + 1) * thetaSize + (i + 1)];
            texcoords[0][numtexts] = textr [(j + 1) * thetaSize + (i + 1)];

            numverts++; numnorms++; numtexts++;

            numfcs++;
            groups[0].numfaces++;
        }

        else
        {
            // Split the quad in two triangles
            // first triangle

#ifndef BUFFER_OBJECT
            Triangle3D *f = &(groups[0].faces[numfcs]);
#else // COMPILE_LIST
            Face3D *f = &(groups[0].faces[numfcs]);
#endif

            f->vertIdx[0] = numverts;
#ifndef BUFFER_OBJECT
            f->normIdx[0] = numnorms;
            f->texcIdx[0] = numtexts;
#endif

            // vertex 0
            vertices    [numverts] = coords[ j      * thetaSize +  i];
            normals     [numnorms] = norms [ j      * thetaSize +  i];
            texcoords[0][numtexts] = textr [ j      * thetaSize +  i];

            numverts++; numnorms++; numtexts++;

            f->vertIdx[1] = numverts;
#ifndef BUFFER_OBJECT
            f->normIdx[1] = numnorms;
            f->texcIdx[1] = numtexts;
#endif

            // vertex 1
            vertices    [numverts] = coords[ j      * thetaSize + (i + 1)];
            normals     [numnorms] = norms [ j      * thetaSize + (i + 1)];
            texcoords[0][numtexts] = textr [ j      * thetaSize + (i + 1)];

            numverts++; numnorms++; numtexts++;

            f->vertIdx[2] = numverts;
#ifndef BUFFER_OBJECT
            f->normIdx[2] = numnorms;
            f->texcIdx[2] = numtexts;
#endif

            // vertex 2
            vertices    [numverts] = coords[(j + 1) * thetaSize + (i + 1)];
            normals     [numnorms] = norms [(j + 1) * thetaSize + (i + 1)];
            texcoords[0][numtexts] = textr [(j + 1) * thetaSize + (i + 1)];

            numverts++; numnorms++; numtexts++;

            numfcs++;
            groups[0].numfaces++;

            f = &(groups[0].faces[numfcs]);

            // second triangle

            f->vertIdx[0] = numverts;
#ifndef BUFFER_OBJECT
            f->normIdx[0] = numnorms;
            f->texcIdx[0] = numtexts;
#endif

            // vertex 0
            vertices    [numverts] = coords[ j      * thetaSize +  i];
            normals     [numnorms] = norms [ j      * thetaSize +  i];
            texcoords[0][numtexts] = textr [ j      * thetaSize +  i];

            numverts++; numnorms++; numtexts++;

            f->vertIdx[1] = numverts;
#ifndef BUFFER_OBJECT
            f->normIdx[1] = numnorms;
            f->texcIdx[1] = numtexts;
#endif

            // vertex 2
            vertices    [numverts] = coords[(j + 1) * thetaSize + (i + 1)];
            normals     [numnorms] = norms [(j + 1) * thetaSize + (i + 1)];
            texcoords[0][numtexts] = textr [(j + 1) * thetaSize + (i + 1)];

            numverts++; numnorms++; numtexts++;

            f->vertIdx[2] = numverts;
#ifndef BUFFER_OBJECT
            f->normIdx[2] = numnorms;
            f->texcIdx[2] = numtexts;
#endif

            // vertex 3
            vertices    [numverts] = coords[(j + 1) * thetaSize +  i];
            normals     [numnorms] = norms [(j + 1) * thetaSize +  i];
            texcoords[0][numtexts] = textr [(j + 1) * thetaSize +  i];

            numverts++; numnorms++; numtexts++;

            numfcs++;
            groups[0].numfaces++;
        }
    }

    EAZD_ASSERTALWAYS (numfaces == numfcs);
} // end of createGeometry

