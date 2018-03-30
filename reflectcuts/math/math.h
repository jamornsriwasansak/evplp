#pragma once

#include "common/reflectcuts.h"

#include <iostream>
#include <algorithm>


namespace Math
{
	// invpi was computed with https://www.wolframalpha.com/input/?i=1%2Fpi
	// according to https://www.jpl.nasa.gov/edu/news/2016/3/16/how-many-decimals-of-pi-do-we-really-need/
	// we should have enough digits to land on the moon.
	const Float Pi = (Float)(3.141592653589793238462643383279502884197169399375105820974);
	const Float InvPi = (Float)(0.318309886183790671537767526745028724068919291480912897495);
	const Float Epsilon = (Float)(1e-6);

	inline size_t MaxExtent(const Vec3 & v)
	{
		if (v.x >= v.y) 
		{
			if (v.x >= v.z) { return 0; }
			else { return 2; }
		}
		else // (y > x)
		{
			if (v.y >= v.z) { return 1; }
			else { return 0; }
		}
		assert(false && "code shouldn't have reached this point");
	}

	template<typename Scalar> inline Scalar Clamp(const Scalar a, const Scalar lower, const Scalar upper)
	{
		return std::max(lower, std::min(upper, a));
	}

	inline int PositiveMod(const int32_t a, const int32_t b)
	{
		int r = a % b;
		return (r < 0) ? r + b : r;
	}

	template <typename Scalar> inline int FloorToInt(Scalar p)
	{
		return (int)std::floor(p);
	}

	// normal must be normalized
	inline void ComputeOrthonormalBasis(Vec3 * xBasis, Vec3 * yBasis, const Vec3 & zBasis)
	{
		Float sign = std::copysign((Float)(1.0), zBasis.z);
		const Float a = -1.0f / (sign + zBasis.z);
		const Float b = zBasis.x * zBasis.y * a;
		*xBasis = Vec3(1.0f + sign * zBasis.x * zBasis.x * a, sign * b, -sign * zBasis.x);
		*yBasis = Vec3(b, sign + zBasis.y * zBasis.y * a, -zBasis.y);
	}

	inline Mat3 ComputeOrthonormalMatrix(const glm::vec3 & zBasis)
	{
		Mat3 result;
		ComputeOrthonormalBasis(&result[0], &result[1], zBasis);
		result[2] = zBasis;
		return result;
	}

	// find the tightest cone that can perfectly bound 2 given cones
	void MergeCone(Vec3 * resultDir, Float * resultHalfAngle, const Vec3 & aDir, const Float aHalfAngle, const Vec3 & bDir, const Float bHalfAngle);
};

// hash function for glm
namespace std
{
	template <>
	struct hash<glm::vec3>
	{
		size_t operator()(const glm::vec3 &in) const { return std::hash<float>()(in.x) ^ std::hash<float>()(in.y) ^ std::hash<float>()(in.z); }
	};

	template <>
	struct hash<glm::vec4>
	{
		size_t operator()(const glm::vec4 &in) const { return std::hash<float>()(in.x) ^ std::hash<float>()(in.y) ^ std::hash<float>()(in.z) ^ std::hash<float>()(in.a); }
	};

	template <>
	struct hash<glm::ivec3>
	{
		size_t operator()(const glm::ivec3 &in) const { return std::hash<int>()(in.x) ^ std::hash<int>()(in.y) ^ std::hash<int>()(in.z); }
	};
};

inline std::ostream& operator<<(std::ostream & os, const glm::uvec3 & vec)
{
	os << glm::to_string(vec);
	return os;
}

inline std::ostream& operator<<(std::ostream & os, const glm::ivec3 & vec)
{
	os << glm::to_string(vec);
	return os;
}

inline std::ostream& operator<<(std::ostream & os, const glm::vec2 & vec)
{
	os << glm::to_string(vec);
	return os;
}

inline std::ostream& operator<<(std::ostream & os, const glm::vec3 & vec)
{
	os << glm::to_string(vec);
	return os;
}

inline std::ostream& operator<<(std::ostream & os, const glm::dvec3 & vec)
{
	os << glm::to_string(vec);
	return os;
}