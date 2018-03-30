#include "shapes/trianglemesh.h"

#include "shapes/trianglemesh.h"
#include "common/sampler.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/gtc/matrix_inverse.hpp>
#include "math/mapping.h"

Float Triangle::ComputeArea(const Vec3 & a, const Vec3 & b, const Vec3 & c)
{
	Vec3 ab = b - a;
	Vec3 ac = c - a;
	Vec3 abcross = glm::cross(ab, ac);
	return glm::length(abcross) / 2.0f;
}

Aabb Triangle::computeBbox() const
{
	Aabb result;
	result = Aabb::Union(Vec3(this->mTriMeshPtr->mVertices[mVertexIndices[0]]), result);
	result = Aabb::Union(Vec3(this->mTriMeshPtr->mVertices[mVertexIndices[1]]), result);
	result = Aabb::Union(Vec3(this->mTriMeshPtr->mVertices[mVertexIndices[2]]), result);
	return result;
}

bool Triangle::clipAabb(Aabb * resultPtr, Float split1, Float split2, uint8_t dim) const
{
	Float Epsilon = 0.00001f;

	Aabb & result = *resultPtr;
	bool isIsect = false;

	Vec3 p[3];
	p[0] = this->mTriMeshPtr->mVertices[this->mVertexIndices[0]];
	p[1] = this->mTriMeshPtr->mVertices[this->mVertexIndices[1]];
	p[2] = this->mTriMeshPtr->mVertices[this->mVertexIndices[2]];

	Vec3 e[3];
	e[0] = p[1] - p[0];
	e[1] = p[2] - p[1];
	e[2] = p[0] - p[2];

	size_t numInside = 0;

	for (size_t i = 0;i < 3;i++)
	{
		// is in between 2 split plane
		if (split1 - Epsilon <= p[i][dim] && p[i][dim] <= split2 + Epsilon)
		{
			result = Aabb::Union(result, p[i]);
			isIsect = true;
			numInside++;
			//assert(p[i][dim] >= split1 - Epsilon && p[i][dim] <= split2 + Epsilon);
		}
	}

	if (split1 == split2)
	{
		return (numInside >= 2);
	}

	// compute all possible intersection with plane
	for (size_t i = 0;i < 3;i++)
	{
		if (e[i][dim] != 0)
		{
			Float invEidim = 1.0f / e[i][dim];
			Float t = (split1 - p[i][dim]) * invEidim;
			if (t >= 0.0f && t <= 1.0f)
			{
				Vec3 isectPos = p[i] + e[i] * t;
				result = Aabb::Union(result, isectPos);
				isIsect = true;
				numInside++;
				//assert(isectPos[dim] >= split1 - Epsilon && isectPos[dim] <= split2 + Epsilon);
			}

			t = (split2 - p[i][dim]) * invEidim;
			if (t >= 0.0f && t <= 1.0f)
			{
				Vec3 isectPos = p[i] + e[i] * t;
				result = Aabb::Union(result, isectPos);
				isIsect = true;
				numInside++;
				//assert(isectPos[dim] >= split1 - Epsilon && isectPos[dim] <= split2 + Epsilon);
			}
		}
	}

	/*if (isIsect)
	{
	assert(split1 - Epsilon <= result.pMin[dim]);
	assert(split2 + Epsilon >= result.pMax[dim]);
	}*/
	return isIsect;
}

std::vector<shared_ptr<TriangleMesh>> TriangleMesh::LoadMeshes(const std::string & filepath, const bool forRealtime)
{
	Assimp::Importer importer;
	unsigned int aiProcesses = aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices;
	if (forRealtime) { aiProcesses = aiProcesses | aiProcessPreset_TargetRealtime_Fast; }
	const aiScene * scene = importer.ReadFile(filepath.c_str(), aiProcesses);

	if (scene == nullptr) {
		std::cerr << "Impossible to load the scene: " << filepath << "\n";
		assert(false);
	}
	
	std::vector<shared_ptr<TriangleMesh>> result(scene->mNumMeshes);
	for (size_t i = 0;i < scene->mNumMeshes;i++)
	{
		auto aiSceneMesh = scene->mMeshes[i];
		result[i] = make_shared<TriangleMesh>();

		TriangleMesh &triangleMesh = *result[i];

		// copy vertices data
		size_t numVertices = aiSceneMesh->mNumVertices;

		triangleMesh.mVertices = std::vector<glm::vec3>(aiSceneMesh->mNumVertices);
		triangleMesh.mNormals = std::vector<glm::vec3>(aiSceneMesh->mNumVertices);
		triangleMesh.mTexCoords = std::vector<glm::vec2>(aiSceneMesh->mNumVertices);

		for (uint32_t j = 0;j < numVertices;j++)
		{
			for (uint32_t k = 0;k < 3;k++)
			{
				triangleMesh.mNormals[j][k] = aiSceneMesh->mNormals[j][k];
			}

			for (uint32_t k = 0;k < 3;k++)
			{
				triangleMesh.mVertices[j][k] = aiSceneMesh->mVertices[j][k];
			}

			for (uint32_t k = 0;k < 2;k++)
			{
				if (aiSceneMesh->HasTextureCoords(0))
				{
					triangleMesh.mTexCoords[j][k] = aiSceneMesh->mTextureCoords[0][j][k];
				}
				else
				{
					triangleMesh.mTexCoords[j][k] = 0.0f;
				}
			}
		}

		// copy triangle data
		size_t numFaces = aiSceneMesh->mNumFaces;
		triangleMesh.mTriangles = std::vector<Triangle>(aiSceneMesh->mNumFaces);
		for (size_t j = 0;j < numFaces;j++)
		{
			for (size_t k = 0;k < 3;k++)
			{
				triangleMesh.mTriangles[j].mVertexIndices[k] = aiSceneMesh->mFaces[j].mIndices[k];
				triangleMesh.mTriangles[j].mNormalIndices[k] = aiSceneMesh->mFaces[j].mIndices[k];
				triangleMesh.mTriangles[j].mTexCoordIndices[k] = aiSceneMesh->mFaces[j].mIndices[k];
			}

			
			// compute geom normal
			Triangle &tri = triangleMesh.mTriangles[j];
			const glm::vec3 & pos1 = triangleMesh.mVertices[tri.mVertexIndices[0]];
			const glm::vec3 & pos2 = triangleMesh.mVertices[tri.mVertexIndices[1]];
			const glm::vec3 & pos3 = triangleMesh.mVertices[tri.mVertexIndices[2]];

			triangleMesh.mTriangles[j].mGeomNormal = glm::normalize(glm::cross(pos3 - pos2, pos1 - pos2));

			triangleMesh.mTriangles[j].mTriMeshPtr = result[i].get();
		}

		if (forRealtime)
		{
			auto aiMat = scene->mMaterials[aiSceneMesh->mMaterialIndex];
			aiColor3D diffuseColor;
			aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, diffuseColor);
			std::cout << diffuseColor.r << std::endl;
		}

		// compute CDF of triangle area
		{
			// compute each triangle area
			Float sumArea = 0.0f;
			triangleMesh.mAreaCdf = std::vector<Float>(numFaces);
			for (size_t j = 0;j < numFaces;j++)
			{
				int idx[3];
				for (size_t k = 0;k < 3;k++)
				{
					idx[k] = triangleMesh.mTriangles[j].mVertexIndices[k];
				}
				Float area = Triangle::ComputeArea(triangleMesh.mVertices[idx[0]], triangleMesh.mVertices[idx[1]], triangleMesh.mVertices[idx[2]]);
				sumArea += area;
				triangleMesh.mAreaCdf[j] = sumArea;
			}

			// normalize cdf
			for (size_t j = 0;j < numFaces;j++)
			{
				triangleMesh.mAreaCdf[j] /= sumArea;
			}
			triangleMesh._mArea = sumArea;
		}

		triangleMesh.recomputeArea();
	}
	importer.FreeScene();
	return result;
}

void TriangleMesh::applyTransform(const glm::mat4 & transformMatrix)
{
	// recompute vertex
	for (glm::vec3 &vertex : mVertices)
	{
		vertex = transformMatrix * glm::vec4(vertex, 1.0f);
	}

	glm::mat4 transformMatrix_normal = glm::inverseTranspose(transformMatrix);
	// recompute shading normal
	for (glm::vec3 &normal : mNormals)
	{
		normal = transformMatrix_normal * glm::vec4(normal, 0.0f);
		normal = glm::normalize(normal);
	}

	// recompute GeomNormal
	for (Triangle & tri : mTriangles)
	{
		tri.mGeomNormal = transformMatrix_normal * glm::vec4(tri.mGeomNormal, 0.0f);
		tri.mGeomNormal = glm::normalize(tri.mGeomNormal);
	}
	this->recomputeArea();
}

Aabb TriangleMesh::computeBbox() const
{
	Aabb result;
	for (const glm::vec3 &vertex : mVertices) { result = Aabb::Union(result, vertex); }
	return result;
}

void TriangleMesh::refine(std::vector<shared_ptr<Shape>>* refine) const
{
	for (size_t i = 0;i < this->mTriangles.size();i++)
	{
		const Triangle & t = this->mTriangles[i];
		shared_ptr<Triangle> p = make_shared<Triangle>(t);
		refine->push_back(p);
	}
}

void TriangleMesh::samplePosition(Vec3 * position, Vec3 * normal, const Sampler & sampler) const
{
	Vec3 sample = Vec3(sampler.nextVec2(), sampler.nextFloat());

	// using first sample to select triangle
	auto iter = std::lower_bound(this->mAreaCdf.begin(), this->mAreaCdf.end(), sample.x);
	size_t iterIndex = iter - this->mAreaCdf.begin();

	// using second and third sample to select on triangle
	int idx[3];
	for (size_t i = 0;i < 3;i++)
	{
		idx[i] = this->mTriangles[iterIndex].mVertexIndices[i];
	}
	Vec2 baryCentric = Mapping::SquareToTriangle(Vec2(sample.y, sample.z));
	Float b0 = baryCentric.x;
	Float b1 = baryCentric.y;
	Float b2 = 1.0f - b1 - b0;

	*position = this->mVertices[idx[0]] * (float)b0 + this->mVertices[idx[1]] * (float)b1 + this->mVertices[idx[2]] * (float)b2;
	if (normal != nullptr)
	{
		*normal = glm::normalize(this->mNormals[idx[0]] * (float)b0 + this->mNormals[idx[1]] * (float)b1 + this->mNormals[idx[2]]);
	}
}

void TriangleMesh::uploadOpenglBuffer()
{
	mVerticesBuffer = make_shared<OpenglBuffer>();
	mIndicesBuffer = make_shared<OpenglBuffer>();
	mTexCoordsBuffer = make_shared<OpenglBuffer>();

	glNamedBufferData(mVerticesBuffer->mHandle, sizeof(float) * mVertices.size() * 3, &(mVertices[0].x), GL_STATIC_DRAW);

	// list vertex indices
	std::vector<uint32_t> indices;
	for (const Triangle & t : this->mTriangles)
	{
		indices.push_back(t.mVertexIndices[0]);
		indices.push_back(t.mVertexIndices[1]);
		indices.push_back(t.mVertexIndices[2]);
	}
	glNamedBufferData(mIndicesBuffer->mHandle, sizeof(uint32_t) * indices.size(), &indices[0], GL_STATIC_DRAW);

	// list tex coords
	// assume vertexindices == texcoordindices
	assert(mVertices.size() == mTexCoords.size());
	glNamedBufferData(mTexCoordsBuffer->mHandle, sizeof(float) * mTexCoords.size() * 2, &(mTexCoords[0].x), GL_STATIC_DRAW);
}

void TriangleMesh::recomputeArea()
{
	// compute each triangle area
	Float sumArea = 0.0;
	size_t numFaces = this->mTriangles.size();
	this->mAreaCdf = std::vector<Float>(numFaces);
	for (size_t j = 0;j < numFaces;j++)
	{
		int idx[3];
		for (size_t k = 0;k < 3;k++)
		{
			idx[k] = this->mTriangles[j].mVertexIndices[k];
		}
		Float area = Triangle::ComputeArea(this->mVertices[idx[0]], this->mVertices[idx[1]], this->mVertices[idx[2]]);
		sumArea += area;
		this->mAreaCdf[j] = sumArea;
	}

	// normalize cdf
	for (size_t j = 0;j < numFaces;j++)
	{
		this->mAreaCdf[j] /= sumArea;
	}
	this->_mArea = sumArea;
}
