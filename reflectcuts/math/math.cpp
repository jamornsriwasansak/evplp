#include "math.h"

void Math::MergeCone(Vec3 * resultDir, Float * resultHalfAngle, const Vec3 & aDir, const Float aHalfAngle, const Vec3 & bDir, const Float bHalfAngle)
{
	Float cosThetaC = glm::dot(aDir, bDir);
	//assert std::abs(cosThetaC <= 1.0000001f); this sometimes fails...
	if (1.0 - cosThetaC <= 0.0000001f) // 2 vectors share the same direction
	{
		if (resultDir != nullptr) { *resultDir = aDir; }
		*resultHalfAngle = std::max(aHalfAngle, bHalfAngle);
		return;
	}

	if (std::abs(cosThetaC + 1.0f) <= 0.0000001f) // 2 vectors are on the opposite side
	{
		if (resultDir != nullptr) { *resultDir = aDir; }
		*resultHalfAngle = Math::Pi;
		return;
	}

	Float thetaC = std::acos(cosThetaC);
	if (thetaC + aHalfAngle <= bHalfAngle) // cone a is completely inside cone b
	{
		if (resultDir != nullptr) { *resultDir = bDir; }
		*resultHalfAngle = bHalfAngle;
	}
	else if (thetaC + bHalfAngle <= aHalfAngle) // cone b is completely inside cone a
	{
		if (resultDir != nullptr) { *resultDir = aDir; }
		*resultHalfAngle = aHalfAngle;
	}

	Float coneHalfAngle = (thetaC + aHalfAngle + bHalfAngle) / 2.0f;
	*resultHalfAngle = coneHalfAngle;
	if (resultDir == nullptr) { return; }

	Float sinThetaC2 = 1.0f - cosThetaC * cosThetaC;
	assert(sinThetaC2 != 0.0f);

	Float cosDiffA = std::cos(coneHalfAngle - aHalfAngle);
	Float cosDiffB = std::cos(coneHalfAngle - bHalfAngle);

	Float x = (cosDiffA - cosDiffB * cosThetaC) / sinThetaC2;
	Float y = (cosDiffB - cosDiffA * cosThetaC) / sinThetaC2;

	*resultDir = glm::normalize(x * aDir + y * bDir);
}
