#include "Core/Precompiled.h"
#include "Utils/Settings.h"
#include <boost/program_options.hpp>
namespace po = boost::program_options;

bool Settings::load(int argc, char** argv)
{
	po::options_description options("Options");
	options.add_options()

		("help", "")

		("model.path", po::value(&model.path)->default_value("Data/"), "")
		("model.name", po::value(&model.name)->default_value("Greeble_big/Greeble_big.obj"), "")

		("general.windowed", po::value(&general.windowed)->default_value(false), "")
		("general.maxCpuThreadCount", po::value(&general.maxCpuThreadCount)->default_value(0), "")
		("general.cudaDeviceNumber", po::value(&general.cudaDeviceNumber)->default_value(0), "")

		("renderer.type", po::value(&renderer.type)->default_value(1), "")
		("renderer.skip", po::value(&renderer.skip)->default_value(false), "")
		("renderer.imageSamples", po::value(&renderer.imageSamples)->default_value(1), "")
		("renderer.pixelSamples", po::value(&renderer.pixelSamples)->default_value(1), "")

		("image.width", po::value(&image.width)->default_value(1280), "")
		("image.height", po::value(&image.height)->default_value(800), "")
		("image.write", po::value(&image.write)->default_value(true), "")
		("image.fileName", po::value(&image.fileName)->default_value("image.png"), "*.png|*.bmp|*.tga|*hdr|*.bin")

		("camera.px", po::value(&camera.px)->default_value(0), "")
		("camera.py", po::value(&camera.py)->default_value(95), "")
		("camera.pz", po::value(&camera.pz)->default_value(-170), "")
		("camera.tx", po::value(&camera.tx)->default_value(0), "")
		("camera.ty", po::value(&camera.ty)->default_value(95), "")
		("camera.tz", po::value(&camera.tz)->default_value(-169), "")
		("camera.ux", po::value(&camera.ux)->default_value(0), "")
		("camera.uy", po::value(&camera.uy)->default_value(1), "")
		("camera.uz", po::value(&camera.uz)->default_value(0), "")
		("camera.fov", po::value(&camera.fov)->default_value(90), "");
		//camera.SetToWorld(Vec3f(0,95,-170), Vec3f(0,95,-169), Vec3f(0,1,0));

	std::ifstream iniFile("cudatracerlib.ini");
	po::variables_map vm;

	try
	{
		//find command line opetions.
		po::store(po::parse_command_line(argc, argv, options), vm);
		//find configurable file options
		po::store(po::parse_config_file(iniFile, options), vm);
		//store in vm
		po::notify(vm);
	}
	catch (const po::error& e)
	{
		std::cout << "Command line / settings file parsing failed: " << e.what() << std::endl;
		std::cout << "Try '--help' for list of valid options" << std::endl;

		return false;
	}

	if(vm.count("help"))
	{
		printf("\033[32;49;2m");
		std::cout << options;
		printf("\033[0m");
		return false;
	}

	return true;
}
