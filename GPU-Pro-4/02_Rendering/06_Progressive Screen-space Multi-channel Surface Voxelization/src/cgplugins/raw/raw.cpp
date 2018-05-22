
#include "raw.h"

#ifdef WIN32
#pragma warning (disable : 4996)
#endif

void rawMesh3D::readFormat (const char * filename)
{
}

void rawMesh3D::writeFormat (const char * filename)
{
    unsigned long i, j, numfaces = 0;
#ifdef BUFFER_OBJECT
    Face3D *t;
#else // COMPILE_LIST
    Triangle3D *t;
#endif

    FILE *fp = fopen (filename, "w");

    EAZD_ASSERTALWAYS (fp);

    // dump in raw triangle format
    for (i = 0; i < numgroups; i++)
        numfaces += groups[i].numfaces;

    printf ("Writing model \"%s\" with %lu faces\n", filename, numfaces);
    for (i = 0; i < numgroups; i++)
        for (j = 0; j < groups[i].numfaces; j++)
        {
            t = &(groups[i].faces[j]);

            fprintf (fp, "% f % f % f\t",
                vertices[t->vertIdx[0]].x,
                vertices[t->vertIdx[0]].y,
                vertices[t->vertIdx[0]].z);
            fprintf (fp, "% f % f % f\t",
                vertices[t->vertIdx[1]].x,
                vertices[t->vertIdx[1]].y,
                vertices[t->vertIdx[1]].z);
            fprintf (fp, "% f % f % f\n",
                vertices[t->vertIdx[2]].x,
                vertices[t->vertIdx[2]].y,
                vertices[t->vertIdx[2]].z);
        }

    fclose (fp);
}

