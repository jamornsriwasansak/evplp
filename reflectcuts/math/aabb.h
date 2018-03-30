#pragma once

#include "math/math.h"
#include <iostream>
#include <numeric>

class Aabb
{
public:
	Vec3 pMin, pMax;

	Aabb():
		pMin(std::numeric_limits<Float>::max()),
		pMax(std::numeric_limits<Float>::lowest())
	{}

	Aabb(const Vec3 & p):
		pMin(p),
		pMax(p)
	{}

	Aabb(const Vec3 & pmin, const Vec3 & pmax):
		pMin(pmin),
		pMax(pmax)
	{}

	inline static Float DiagonalLength2(const Aabb & a)
	{
		Vec3 diag = glm::max(a.pMax - a.pMin, Vec3(0.0f));
		return glm::dot(diag, diag);
	}

	inline static Aabb Union(const Aabb & a, const Vec3 & b)
	{
		return Aabb(glm::min(a.pMin, b), glm::max(a.pMax, b));
	}

	inline static Aabb Union(const Aabb & a, const Aabb & b)
	{
		return Aabb(glm::min(a.pMin, b.pMin), glm::max(a.pMax, b.pMax));
	}

	inline static Aabb Intersect(const Aabb & a, const Aabb & b)
	{
		return Aabb(glm::max(a.pMin, b.pMin), glm::min(a.pMax, b.pMax));
	}

	inline static Aabb Padding(const Aabb & a, float p)
	{
		return Aabb(a.pMin - Vec3(p), a.pMax + Vec3(p));
	}

	inline static bool IsInside(const Vec3 & p, const Aabb & a)
	{
		return p.x >= a.pMin.x && p.y >= a.pMin.y && p.z >= a.pMin.z &&
			p.x <= a.pMax.x && p.y <= a.pMax.y && p.z <= a.pMax.z;
	}

	inline size_t maxExtent() const
	{
		Vec3 v = pMax - pMin;
		return Math::MaxExtent(v);
		assert(false && "code shouldn't reach this point");
	}

	inline Vec3 computeCentroid() const
	{
		return (Float)0.5 * (pMin + pMax);
	}

	inline Float surfaceArea() const
	{
		Vec3 edges = glm::clamp(pMax - pMin, Vec3(0.0f), Vec3(std::numeric_limits<Float>::infinity()));
		Float result = 2.0f * ((edges.x * edges.y) + (edges.y * edges.z) + (edges.x * edges.z));
		return (result > 0) ? result : 0;
	}

	inline Float volume() const
	{
		Vec3 edges = glm::clamp(pMax - pMin, Vec3(0.0f), Vec3(std::numeric_limits<Float>::infinity()));
		return edges.x * edges.y * edges.z;
	}

	// copied from http://dev.theomader.com/transform-bounding-boxes/
	inline static Aabb Transform(const Aabb & a, const Mat3 & mat)
	{
		Vec3 mm(0.0f), nn(0.0f);
		for (size_t i = 0;i < 3;i++)
		{
			Vec3 m = mat[i] * a.pMin[i];
			Vec3 n = mat[i] * a.pMax[i];
			mm += glm::min(m, n);
			nn += glm::max(m, n);
		}
		return Aabb(mm, nn);
	}

	// same as Transform but surely 100% correct
	inline static Aabb Transform_Exhaust(const Aabb & a, const Mat3 & mat)
	{
		Vec3 corners[8];
		FillWith8Corners(corners, a);
		Aabb result;
		for (size_t i = 0;i < 8;i++)
		{
			result = Aabb::Union(result, mat * corners[i]);
		}
		return result;
	}

	// distance between a point and aabb by User MultiRRomero in stackoverflow
	inline static Float ShortestDistance2(const Aabb & a, const Vec3 & p)
	{
		Float d[3];
		for (size_t i = 0;i < 3;i++)
		{
			Float left = a.pMin[i] - p[i];
			Float right = p[i] - a.pMax[i];
			d[i] = std::max((Float)0.0, std::max(left, right));
		}
		return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
	}

	inline static void FillWith8Corners(Vec3 points[], const Aabb & a)
	{
		const Vec3 & x = a.pMin;
		const Vec3 & y = a.pMax;
		points[0] = x;
		points[1] = Vec3(y[0], x[1], x[2]);
		points[2] = Vec3(x[0], y[1], x[2]);
		points[3] = Vec3(x[0], x[1], y[2]);
		points[4] = Vec3(x[0], y[1], y[2]);
		points[5] = Vec3(y[0], x[1], y[2]);
		points[6] = Vec3(y[0], y[1], x[2]);
		points[7] = y;
	}

	// refers to Lightcuts: A Scalable Approach to Illumination eq. 4
	static inline Float MaxCosBound(const Aabb & bound)
	{
		Float maxPz = bound.pMax.z;
		Float denominator2;
		if (maxPz >= 0)
		{
			// same to Aabb::Distance2
			Float absMinPx = std::max((Float)(0.0), std::max(-bound.pMax.x, bound.pMin.x));
			Float absMinPy = std::max((Float)(0.0), std::max(-bound.pMax.y, bound.pMin.y));
			Float minPx2 = absMinPx * absMinPx;
			Float minPy2 = absMinPy * absMinPy;
			Float maxPz2 = maxPz * maxPz;

			denominator2 = minPx2 + minPy2 + maxPz2;
		}
		else
		{
			Float absMaxPx = std::max(bound.pMax.x, -bound.pMin.x);
			Float absMaxPy = std::max(bound.pMax.y, -bound.pMin.y);
			Float maxPx2 = absMaxPx * absMaxPx;
			Float maxPy2 = absMaxPy * absMaxPy;
			Float maxPz2 = maxPz * maxPz;

			denominator2 = maxPx2 + maxPy2 + maxPz2;
		}

		if (denominator2 == 0.0f) { return 1.0f; }  // return highest possible cos value; 
		return maxPz / std::sqrt(denominator2);
	}

	inline friend std::ostream & operator<< (std::ostream & out, const Aabb & vec)
	{
		return out << "Aabb(" << vec.pMin << "," << vec.pMax << ")";
	}
};