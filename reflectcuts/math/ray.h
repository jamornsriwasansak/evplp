#pragma once

#include <numeric>
#include "math/math.h"

class Ray
{
public:
	// prevent self intersection
	static const Float Epsilon;

	Ray(const Vec3 & origin, const Vec3 & direction):
		origin(origin),
		direction(direction),
		tmin(Ray::Epsilon),
		tmax(std::numeric_limits<Float>::infinity())
	{
	}

	Ray(const Vec3 & origin, const Vec3 & direction, Float mint, Float maxt):
		origin(origin),
		direction(direction),
		tmin(mint),
		tmax(maxt) 
	{
	}

	inline Vec3 t(Float k) { return origin + direction * k; }

	Vec3 origin;
	Vec3 direction;
	Float tmin;
	Float tmax;
};
