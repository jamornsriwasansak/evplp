#pragma once

#include "math/math.h"

class Mapping
{
public:
	static inline Vec3 SphericalToWorld(const Vec2 & thetaPhi)
	{
		const Float & theta = thetaPhi.x;
		const Float & phi = thetaPhi.y;
		const Float sintheta = std::sin(theta);
		const Float costheta = std::cos(theta);
		const Float sinphi = std::sin(phi);
		const Float cosphi = std::cos(phi);
		return Vec3(cosphi * sintheta, sinphi * sintheta, costheta);
	}

	static inline Vec2 WorldToSpherical(const Vec3 & pos)
	{
		const Float phi = std::atan2(pos.y, pos.x);
		const Float numerator = std::sqrt(pos.x * pos.x + pos.y * pos.y);
		const Float theta = std::atan2(numerator, pos.z);
		return Vec2(theta, phi);
	}

	// modified from http://gl.ict.usc.edu/Data/HighResProbes/
	// similar to spherical to world with some swizzle (y is costheta)
	static inline Vec3 PanoramaToWorld(const Vec2 & uv)
	{
		const Float u = uv.x * 2 - 1.0f;
		const Float v = uv.y;
		const Float theta = Math::Pi * v;
		const Float phi = Math::Pi * (u - 1.0f);
		const Float sintheta = std::sin(theta);
		const Float costheta = std::cos(theta);
		const Float sinphi = std::sin(phi);
		const Float cosphi = std::cos(phi);
		return Vec3(sintheta * sinphi, costheta, -sintheta * cosphi);
	}

	// modified from http://gl.ict.usc.edu/Data/HighResProbes/
	static inline Vec2 WorldToPanorama(const Vec3 & dir_Norm)
	{
		return Vec2((1.0f + std::atan2(-dir_Norm.x, dir_Norm.z) * Math::InvPi) * 0.5f, std::acos(dir_Norm.y) * Math::InvPi);
	}

	// Octahedron Environment Maps by Thomas Engelhardt, Carsten Dachsbacher eq (1) and (3). Y and Z axis (positive hemisphere) are swapped
	static inline Vec2 WorldToOctahedron(const Vec3 & dir_Norm)
	{
		const Vec3 pdash = dir_Norm / (std::abs(dir_Norm.x) + std::abs(dir_Norm.y) + std::abs(dir_Norm.z));
		const Float sign = std::copysign((Float)1.0, pdash.z);
		const Float u = (sign * (pdash.x - pdash.y - 1.0f) + 2.0f) / 4.0f;
		const Float v = (pdash.x + pdash.y + 1.0f) / 2.0f;
		return Vec2(u, v);
	}

	// Inverse Octahedron Environment Maps derived from eq (3). Y and Z axis (positive hemisphere) are swapped
	static inline Vec3 OctahedronToWorld(const Vec2 & uv)
	{
		const Float u2 = (uv.x * 4.0f) - 2.0f;
		const Float v2 = (uv.y * 2.0f) - 1.0f;
		const Float sign = std::copysign((Float)1.0, u2);
		const Float u3 = u2 * sign;

		const Float px = (v2 - u3 + 1.0f) / 2.0f;
		const Float py = (v2 + u3 - 1.0f) / 2.0f;
		const Float pz = sign * (std::abs(px) + std::abs(py) - 1.0f);
		return Vec3(px, py, pz);
	}

	static inline Vec3 SquareToSphere(const Vec2 & sample)
	{
		const Float z = 1.0f - 2.0f * sample.y;
		const Float r = std::sqrt(sample.y * (1.0f - sample.y));
		const Float phi = 2.0f * Math::Pi * sample.x; // phi = [0, 2pi)
		const Float cosphi = std::cos(phi);
		const Float sinphi = std::sin(phi);
		return Vec3(2.0f * cosphi * r, 2.0f * sinphi * r, z);
	}

	static inline Vec3 SquareToHemisphere(const Vec2 & sample)
	{
		// z = [0, 1]
		const Float r = std::sqrt(1.0f - sample.x * sample.x);
		const Float phi = 2.0f * Math::Pi * sample.y; // phi = [0, 2pi)
		const Float cosphi = std::cos(phi);
		const Float sinphi = std::sin(phi);
		return Vec3(cosphi * r, sinphi * r, sample.x);
	}

	// rtheta, rphi
	static inline Vec3 SquareToCosineWeightedHemisphere(const Vec2 & sample)
	{
		const Float term1 = std::sqrt(1.0f - sample.x);
		const Float term2 = Math::Pi * 2.0f * sample.y; // phi
		const Float cosphi = std::cos(term2);
		const Float sinphi = std::sin(term2);
		return Vec3(cosphi * term1, sinphi * term1, std::sqrt(sample.x));
	}

	// taken from Total Compendium pg. 19 (34)
	static inline Vec3 SquareToSolidAngle(const Vec2 & sample, const Float halfAngleMax)
	{
		const Float phi = 2.0f * Math::Pi * sample.x;
		const Float z = 1.0f - sample.y * (1.0f - std::cos(halfAngleMax));
		const Float l = std::sqrt(1.0f - z * z);
		const Float cosphi = std::cos(phi);
		const Float sinphi = std::sin(phi);
		return Vec3(cosphi * l, sinphi * l, z);
	}

	// taken from https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle
	static inline Vec2 SquareToTriangle(const Vec2 & sample)
	{
		Vec2 sqrtSample = glm::sqrt(sample);
		Float b1 = (sqrtSample.x * (1.0f - sample.y));
		Float b2 = (sample.y * sqrtSample.x);
		return Vec2(b1, b2);
	}

	// taken from Dave Cline from Peter Shirley from http://psgraphics.blogspot.jp/2011/01/improved-code-for-concentric-map.html
	// and https://github.com/mmp/pbrt-v3/blob/7095acfd8331cdefb03fe0bcae32c2fc9bd95980/src/core/sampling.cpp
	static inline Vec2 SquareToDisk(const Vec2 & sample)
	{
		Vec2 ab = sample * (Float)2.0 - Vec2(1.0f);

		if (ab.x == 0 && ab.y == 0) { return Vec2(0.0f); }

		Vec2 ab2 = ab * ab;
		Float phi, r;
		if (ab2.x > ab2.y) { // use squares instead of absolute values
			r = ab.x;
			phi = (Math::Pi / 4.0f) * (ab.y / ab.x);
		}
		else {
			r = ab.y;
			phi =  (Math::Pi / 2.0f) - (Math::Pi / 4.0f) * (ab.x / ab.y);
		}
		Float cosphi = std::cos(phi);
		Float sinphi = std::sin(phi);

		return r * Vec2(cosphi, sinphi);
	}
};