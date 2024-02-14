//
// Modified from mikktpy to support buffer interface
// Original Copyright (c) 2013 Ambrus Csaszar
//
#include <stdlib.h>
#include "mikktspace.h"


static const int components_per_vert = 3 + 3 + 2;  // xyz, normal-xyz, uv
static const int components_per_tangent = 4;  // tangent + sign
static const int vertices_per_face = 3;

typedef struct 
{
  const float * v;
  float * r;
  long long n;
} TriTangentState;

// Returns the number of faces (triangles/quads) on the mesh to be processed.
int getNumFaces(const SMikkTSpaceContext * pContext);

// Returns the number of vertices on face number iFace
// iFace is a number in the range {0, 1, ..., getNumFaces()-1}
int getNumVerticesOfFace(const SMikkTSpaceContext * pContext, const int iFace);

// returns the position/normal/texcoord of the referenced face of vertex number iVert.
// iVert is in the range {0,1,2} for triangles and {0,1,2,3} for quads.
void getPosition(const SMikkTSpaceContext * pContext, float fvPosOut[], const int iFace, const int iVert);
void getNormal(const SMikkTSpaceContext * pContext, float fvNormOut[], const int iFace, const int iVert);
void getTexCoord(const SMikkTSpaceContext * pContext, float fvTexcOut[], const int iFace, const int iVert);

// either (or both) of the two setTSpace callbacks can be set.
// The call-back m_setTSpaceBasic() is sufficient for basic normal mapping.

// This function is used to return the tangent and fSign to the application.
// fvTangent is a unit length vector.
// For normal maps it is sufficient to use the following simplified version of the bitangent which is generated at pixel/vertex level.
// bitangent = fSign * cross(vN, tangent);
// Note that the results are returned unindexed. It is possible to generate a new index list
// But averaging/overwriting tangent spaces by using an already existing index list WILL produce INCRORRECT results.
// DO NOT! use an already existing index list.
void setTSpaceBasic(const SMikkTSpaceContext * pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert);


int getNumFaces(const SMikkTSpaceContext * pContext)
{
  TriTangentState *s = (TriTangentState*)pContext->m_pUserData;
  int ret = s->n / vertices_per_face;
  // cout << "GetNumFaces: " << ret << endl;
  return ret;
}

int getNumVerticesOfFace(const SMikkTSpaceContext * pContext, const int iFace)
{
  // cout << "GetVerticesPerFace: " << vertices_per_face << endl;
  return vertices_per_face;
}

void getPosition(const SMikkTSpaceContext * pContext, float fvPosOut[], const int iFace, const int iVert)
{
  TriTangentState *s = (TriTangentState*)pContext->m_pUserData;
  int idx = iFace * vertices_per_face * components_per_vert + iVert * components_per_vert;
  // cout << "GetPosition: " << idx << endl;
  fvPosOut[0] = s->v[idx + 0];
  fvPosOut[1] = s->v[idx + 1];
  fvPosOut[2] = s->v[idx + 2];
}

void getNormal(const SMikkTSpaceContext * pContext, float fvNormOut[], const int iFace, const int iVert)
{
  TriTangentState *s = (TriTangentState*)pContext->m_pUserData;
  int idx = iFace * vertices_per_face * components_per_vert + iVert * components_per_vert;
  fvNormOut[0] = s->v[idx + 3];
  fvNormOut[1] = s->v[idx + 4];
  fvNormOut[2] = s->v[idx + 5];
}

void getTexCoord(const SMikkTSpaceContext * pContext, float fvTexcOut[], const int iFace, const int iVert)
{
  TriTangentState *s = (TriTangentState*)pContext->m_pUserData;
  int idx = iFace * vertices_per_face * components_per_vert + iVert * components_per_vert;
  fvTexcOut[0] = s->v[idx + 6];
  fvTexcOut[1] = s->v[idx + 7];
}

void setTSpaceBasic(const SMikkTSpaceContext * pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert)
{
  TriTangentState *s = (TriTangentState*)pContext->m_pUserData;
  int idx = iFace * vertices_per_face * components_per_tangent + iVert * components_per_tangent;

  // cout << "SetTspace (" << iFace << "," << iVert << ") ";
  // for (int i = 0; i < 3; i++)
  //   cout << (i>0 ? "," : "") << fvTangent[i];
  // cout << ";   ";
  // for (int i = 0; i < 3; i++)
  //   cout << (i>0 ? "," : "") << fvBiTangent[i];
  // cout << endl;
  // cout << "idx: " << idx << endl;

  s->r[idx + 0] = fvTangent[0];
  s->r[idx + 1] = fvTangent[1];
  s->r[idx + 2] = fvTangent[2];
  s->r[idx + 3] = fSign;
}

void compute_tri_tangents(const float * v, float * r, long long n)
{
  SMikkTSpaceInterface mikktInterface;
  mikktInterface.m_getNumFaces = getNumFaces;
  mikktInterface.m_getNumVerticesOfFace = getNumVerticesOfFace;
  mikktInterface.m_getPosition = getPosition;
  mikktInterface.m_getNormal = getNormal;
  mikktInterface.m_getTexCoord = getTexCoord;
  mikktInterface.m_setTSpaceBasic = setTSpaceBasic;
  mikktInterface.m_setTSpace = NULL;

  TriTangentState s = { v, r, n };

  SMikkTSpaceContext mikktContext;
  mikktContext.m_pInterface = &mikktInterface;
  mikktContext.m_pUserData = &s;

  genTangSpaceDefault(&mikktContext);
}
