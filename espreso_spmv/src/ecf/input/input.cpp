
#include "input.h"

#include "config/configuration.hpp"

using namespace espreso;

ClippingBox::ClippingBox()
{
	apply = false;
	REGISTER(apply, ECFMetaData()
			.setdescription({ "Clip the input according to the box." })
			.setdatatype({ ECFDataType::BOOL }));

	min[0] = min[1] = min[2] = max[0] = max[1] = max[2] = 0;
	ecfdescription->registerParameter("min_x", min[0], ECFMetaData().setdescription({ "MIN.X" }).setdatatype({ ECFDataType::FLOAT }));
	ecfdescription->registerParameter("min_y", min[1], ECFMetaData().setdescription({ "MIN.Y" }).setdatatype({ ECFDataType::FLOAT }));
	ecfdescription->registerParameter("min_z", min[2], ECFMetaData().setdescription({ "MIN.Z" }).setdatatype({ ECFDataType::FLOAT }));
	ecfdescription->registerParameter("max_x", max[0], ECFMetaData().setdescription({ "MAX.X" }).setdatatype({ ECFDataType::FLOAT }));
	ecfdescription->registerParameter("max_y", max[1], ECFMetaData().setdescription({ "MAX.Y" }).setdatatype({ ECFDataType::FLOAT }));
	ecfdescription->registerParameter("max_z", max[2], ECFMetaData().setdescription({ "MAX.Z" }).setdatatype({ ECFDataType::FLOAT }));
}

InputConfiguration::InputConfiguration()
{
	path = ".";
	REGISTER(path, ECFMetaData()
			.setdescription({ "Path" })
			.setdatatype({ ECFDataType::STRING }));

	format = FORMAT::ANSYS_CDB;
	REGISTER(format, ECFMetaData()
			.setdescription({ "An input data type." })
			.setdatatype({ ECFDataType::OPTION })
			.addoption(ECFOption().setname("ANSYS_CDB").setdescription("Ansys WorkCDB format."))
			.addoption(ECFOption().setname("OpenFOAM").setdescription("OpenFOAM format."))
			.addoption(ECFOption().setname("ABAQUS").setdescription("ABAQUS format."))
			.addoption(ECFOption().setname("XDMF").setdescription("XDMF format."))
			.addoption(ECFOption().setname("ENSIGHT").setdescription("Ensigh Gold format."))
			.addoption(ECFOption().setname("VTK_LEGACY").setdescription("VTK Legacy format."))
			.addoption(ECFOption().setname("NETGEN").setdescription("Neutral Netgen format."))
			.addoption(ECFOption().setname("NEPER").setdescription("Neper Msh format."))
			.addoption(ECFOption().setname("GMSH").setdescription("GMSH mesh generator."))
			.addoption(ECFOption().setname("NGLIB").setdescription("NGLib mesh generator.")));

	REGISTER(clipping_box, ECFMetaData().setdescription({ "Clipping box." }));

	omit_midpoints = false;
	REGISTER(omit_midpoints, ECFMetaData()
			.setdescription({ "All mid-points within elements are omitted." })
			.setdatatype({ ECFDataType::BOOL }));

	insert_midpoints = false;
//	REGISTER(insert_midpoints, ECFMetaData()
//			.setdescription({ "Insert mid-points to all linear elements." })
//			.setdatatype({ ECFDataType::BOOL }));

	omit_face_sets = false;
	REGISTER(omit_face_sets, ECFMetaData()
			.setdescription({ "Do not read face sets." })
			.setdatatype({ ECFDataType::BOOL }));

	keep_material_sets = false;
	REGISTER(keep_material_sets, ECFMetaData()
			.setdescription({ "Keep material sets" })
			.setdatatype({ ECFDataType::BOOL }));

	convert_database = false;
	REGISTER(convert_database, ECFMetaData()
			.setdescription({ "Convert database" })
			.setdatatype({ ECFDataType::BOOL }));

	duplication_tolerance = 1e-8;
	REGISTER(duplication_tolerance, ECFMetaData()
			.setdescription({ "Tolerance for merging nodes according to coordinates." })
			.setdatatype({ ECFDataType::FLOAT }));

	insert_orientation = false;
//	REGISTER(insert_orientation, ECFMetaData()
//			.setdescription({ "Insert orientation (only for NEPER)." })
//			.setdatatype({ ECFDataType::BOOL }));

	loader = LOADER::MPI;
	REGISTER(loader, ECFMetaData()
			.setdescription({ "A type of used function for loading data." })
			.setdatatype({ ECFDataType::OPTION })
			.addoption(ECFOption().setname("MPI").setdescription("Use MPI_File_read_at."))
			.addoption(ECFOption().setname("MPI_COLLECTIVE").setdescription("Use MPI_File_read_at_all."))
			.addoption(ECFOption().setname("POSIX").setdescription("Use POSIX API.")));

	stripe_size = 1024 * 1024;
	REGISTER(stripe_size, ECFMetaData()
			.setdescription({ "Stripe size of the input file." })
			.setdatatype({ ECFDataType::NONNEGATIVE_INTEGER }));

	third_party_scalability_limit = 768;
	REGISTER(third_party_scalability_limit, ECFMetaData()
			.setdescription({ "Maximum number of MPI processes used for non-scalable routines." })
			.setdatatype({ ECFDataType::NONNEGATIVE_INTEGER }));

	savedomain = false;
	REGISTER(savedomain, ECFMetaData()
			.setdescription({ "saving domain xyz." })
			.setdatatype({ ECFDataType::BOOL }));

	loaddomain = false;
	REGISTER(loaddomain, ECFMetaData()
			.setdescription({ "loading domain xyz." })
			.setdatatype({ ECFDataType::BOOL }));

	geomparmetis = false;
	REGISTER(geomparmetis, ECFMetaData()
			.setdescription({ "geom parmetis." })
			.setdatatype({ ECFDataType::BOOL }));

	scancluster = false;
	REGISTER(scancluster, ECFMetaData()
			.setdescription({ "scan cluster." })
			.setdatatype({ ECFDataType::BOOL }));
			
	use_device = false;
	REGISTER(use_device, ECFMetaData()
			.setdescription({ "use_device" })
			.setdatatype({ ECFDataType::BOOL }));

	device_count = 1;
	REGISTER(device_count, ECFMetaData()
			.setdescription({ "device_count" })
			.setdatatype({ ECFDataType::NONNEGATIVE_INTEGER }));

	device_stream_count = 1;
	REGISTER(device_stream_count, ECFMetaData()
			.setdescription({ "device_stream_count" })
			.setdatatype({ ECFDataType::NONNEGATIVE_INTEGER }));

	use_device_async = 0;
	REGISTER(use_device_async, ECFMetaData()
			.setdescription({ "use_device_async" })
			.setdatatype({ ECFDataType::NONNEGATIVE_INTEGER }));

	use_device_print = false;
	REGISTER(use_device_print, ECFMetaData()
			.setdescription({ "use_device_print" })
			.setdatatype({ ECFDataType::BOOL }));
	
	save_mat = false;
	REGISTER(save_mat, ECFMetaData()
			.setdescription({ "save_mat" })
			.setdatatype({ ECFDataType::BOOL }));

	use_ddot_opt = 0;
	REGISTER(use_ddot_opt, ECFMetaData()
			.setdescription({ "use_ddot_opt" })
			.setdatatype({ ECFDataType::NONNEGATIVE_INTEGER }));

	use_mpi_balance = false;
	REGISTER(use_mpi_balance, ECFMetaData()
			.setdescription({ "use_mpi_balance" })
			.setdatatype({ ECFDataType::BOOL }));

	ecfdescription->addSeparator();

	REGISTER(node_regions, ECFMetaData()
			.setdescription({ "List of transformations", "Transformation" })
			.setdatatype({ ECFDataType::ELEMENTS_REGION })
			.setpattern({ "ALL_ELEMENTS" }));

	REGISTER(transformations, ECFMetaData()
			.setdescription({ "List of transformations", "Transformation" })
			.setdatatype({ ECFDataType::ELEMENTS_REGION })
			.setpattern({ "ALL_ELEMENTS" }));

	REGISTER(decomposition, ECFMetaData()
			.setdescription({ "Domains decomposition settings." }));

	REGISTER(generation, ECFMetaData()
			.setdescription({ "Mesh generation settings." }));

	REGISTER(contact_interfaces, ECFMetaData()
				.setdescription({ "List of contact interfaces", "Contact interface definition" })
				.setdatatype({ ECFDataType::STRING })
				.setpattern({ "CONTACT_1" }));
}

