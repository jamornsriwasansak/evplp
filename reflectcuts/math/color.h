#pragma once

#include "math/math.h"

namespace Color
{
	const Vec3 Red = Vec3(1.0f, 0.0f, 0.0f);
	const Vec3 Blue = Vec3(0.0f, 1.0f, 0.0f);
	const Vec3 Green = Vec3(0.0f, 0.0f, 1.0f);

	inline Float Luminance(const Vec3 & color)
	{
		return glm::dot(Vec3(0.2126, 0.7152, 0.0722), color);
	}

	inline Float Hue2Rgb(Vec3 hue) {
		if (hue.z < 0.0f) hue.z += 1.0f;
		if (hue.z > 1.0f) hue.z -= 1.0f;
		if ((6.0f * hue.z) < 1.0f) return (hue.x + (hue.y - hue.x) * 6.0f * hue.z);
		if ((2.0f * hue.z) < 1.0f) return hue.y;
		if ((3.0f * hue.z) < 2.0f) return (hue.x + (hue.y - hue.x) * ((2.0f / 3.0f) - hue.z) * 6.0f);
		return hue.x;
	}

	// ref https://stackoverflow.com/questions/2353211/hsl-to-rgb-color-conversion
	inline Vec3 Hsl2Rgb(const Vec3 & hsl) {
		if (hsl[1] == 0)
		{
			return Vec3(hsl[2]);
		}
		else
		{
			Float v1, v2;
			Float hue = hsl[0];

			v2 = (hsl[2] < 0.5f) ? (hsl[2] * (1.0f + hsl[0])) : ((hsl[1] + hsl[2]) - (hsl[1] * hsl[2]));
			v1 = 2.0f * hsl[2] - v2;

			return Vec3(
				Hue2Rgb(Vec3(v1, v2, hue + (1.0f / 3.0f))),
				Hue2Rgb(Vec3(v1, v2, hue)),
				Hue2Rgb(Vec3(v1, v2, hue - (1.0f / 3.0f)))
			);
		}
	}

	// ref http://www.rapidtables.com/convert/color/rgb-to-hsl.htm
	inline Vec3 Rgb2Hsl(const Vec3 &rgb)
	{
		Float cmax = std::max(std::max(rgb[0], rgb[1]), rgb[2]);
		Float cmin = std::min(std::min(rgb[0], rgb[1]), rgb[2]);

		Float delta = cmax - cmin;

		if (delta == 0.0f)
		{
			Float h = 0.0f, s = 0.0f, l = (cmax + cmin) / 2.0f;
			return Vec3(h, s, l);
		}
		else
		{
			Float l = (cmax + cmin) / 2.0f;
			Float s = delta / (1.0f - std::abs(2.0f * l - 1.0f));
			Float h = 0.0f;
			if (cmax == rgb.r)
			{
				h = (rgb.g - rgb.b) / delta + (rgb.g < rgb.b ? 6 : 0);
			}
			else if (cmax == rgb.g)
			{
				h = (rgb.b - rgb.r) / delta + 2.0f;
			}
			else if (cmax == rgb.b)
			{
				h = (rgb.r - rgb.g) / delta + 4.0f;
			}
			h /= 6;
			return Vec3(h, s, l);
		}
	}

	// ref https://stackoverflow.com/questions/12875486/what-is-the-algorithm-to-create-colors-for-a-heatmap
	inline Vec3 Heat(const Float temperature)
	{
		Float h = ((1.0f - temperature) * 240.0f) / 360.0f;
		return Hsl2Rgb(Vec3(h, 1.0f, 0.5f));
	}
};