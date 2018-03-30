#pragma once

#include "common/reflectcuts.h"
#include "common/shape.h"

#include <vector>
#include "math/math.h"
#include "math/aabb.h"

#include "opengl/buffer.h"

#include "optix.h"
#include "optixu/optixu.h"
#include "optixu/optixpp_namespace.h"
#include "optix_gl_interop.h"

class TriangleMesh;
class Triangle : public Shape
{
public:
	static Float ComputeArea(const Vec3 & p1, const Vec3 & p2, const Vec3 & p3);

	Aabb computeBbox() const override;

	bool clipAabb(Aabb * resultPtr, Float split1, Float split2, uint8_t dim) const override;
	inline bool canIntersect() const override { return true; }

	// don't replace this with weak_ptr. performance reasons.
	TriangleMesh * mTriMeshPtr;
	uint32_t mVertexIndices[3];
	uint32_t mTexCoordIndices[3];
	uint32_t mNormalIndices[3];
	glm::vec3 mGeomNormal;
};

struct OptixMeshBuffer
{
	optix::GeometryInstance mGeometryInstance;
	optix::Geometry mGeometry;
	optix::Buffer mIndices;
	optix::Buffer mVertices;
	optix::Buffer mNormals;
	// we don't need this yet
	//optix::Buffer mTexCoords;
};

// this is meant for opengl stuff

class TriangleMesh : public Shape
{
public:
	static std::vector<shared_ptr<TriangleMesh>> LoadMeshes(const std::string & filepath, bool forRealtime = false);

	void applyTransform(const glm::mat4 & transformMatrix);
	Aabb computeBbox() const override;
	void refine(std::vector<shared_ptr<Shape>> * refine) const override;
	void samplePosition(Vec3 * position, Vec3 * direction, const Sampler & sampler) const override;
	void uploadOpenglBuffer();

	void uploadOptix(optix::Context context, optix::Program meshIntersect, optix::Program bboxIntersect, optix::Material material)
	{
		mOptix.mIndices = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, this->mTriangles.size());
		mOptix.mVertices = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, this->mVertices.size());
		mOptix.mNormals = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, this->mVertices.size());
		//mOptixBuffer.mTexCoords = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, this->mVertices.size());

		// map buffers
		int32_t* indices = reinterpret_cast<int32_t*>(mOptix.mIndices->map());
		float* vertices = reinterpret_cast<float*>(mOptix.mVertices->map());
		float* normals = reinterpret_cast<float*>(mOptix.mNormals->map());

		// fill mapped buffers
		{
			// fill indices
			for (size_t i = 0;i < mTriangles.size();i++)
			{
				indices[i * 3 + 0] = mTriangles[i].mVertexIndices[0];
				indices[i * 3 + 1] = mTriangles[i].mVertexIndices[1];
				indices[i * 3 + 2] = mTriangles[i].mVertexIndices[2];
			}

			for (size_t i = 0;i < mVertices.size();i++)
			{
				vertices[i * 3 + 0] = mVertices[i].x;
				vertices[i * 3 + 1] = mVertices[i].y;
				vertices[i * 3 + 2] = mVertices[i].z;

				normals[i * 3 + 0] = mNormals[i].x;
				normals[i * 3 + 0] = mNormals[i].x;
				normals[i * 3 + 0] = mNormals[i].x;
			}
		}

		mOptix.mGeometry = context->createGeometry();
		mOptix.mGeometry->setPrimitiveCount(mTriangles.size());
		mOptix.mGeometry->setIntersectionProgram(meshIntersect);
		mOptix.mGeometry->setBoundingBoxProgram(bboxIntersect);
		mOptix.mGeometry["vertexBuffer"]->setBuffer(mOptix.mVertices);
		mOptix.mGeometry["normalBuffer"]->setBuffer(mOptix.mNormals);
		mOptix.mGeometry["indexBuffer"]->setBuffer(mOptix.mIndices);

		// unmap buffers
		mOptix.mIndices->unmap();
		mOptix.mVertices->unmap();
		mOptix.mNormals->unmap();

		mOptix.mGeometryInstance = context->createGeometryInstance(mOptix.mGeometry, &material, &material + 1);
	}

	inline Float mArea() const override { return _mArea; }

	//friend class EmbreeAccel;
	//friend class Triangle;
//private:

	void recomputeArea();

	std::vector<glm::vec3> mVertices;
	std::vector<glm::vec3> mNormals;
	std::vector<glm::vec2> mTexCoords;
	std::vector<Triangle> mTriangles;

	shared_ptr<OpenglBuffer> mVerticesBuffer;
	shared_ptr<OpenglBuffer> mIndicesBuffer;
	shared_ptr<OpenglBuffer> mTexCoordsBuffer;

	OptixMeshBuffer mOptix;
	
	std::vector<Float> mAreaCdf;
	Float _mArea;
};
