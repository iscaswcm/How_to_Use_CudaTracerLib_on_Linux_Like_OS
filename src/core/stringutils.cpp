#include "tinyformat/tinyformat.h"
#include "core/stringutils.h"

#include "core/precompiled.h"

std::string StringUtils::humanizeNumber(double value, bool usePowerOfTwo)
{
    const char* postfixes[] = { "", " k", " M", " G", " T", " P", " E", " Z", " Y" };
	const double divider = usePowerOfTwo ? 1024.0 : 1000.0;

    for (auto& postfix : postfixes)
	{
		if(value < divider)
            return tfm::format("%.2f%s", value, postfix);
		else
			value /= divider;
	}

    return tfm::format("%.2f%s", value, " Y");
}
