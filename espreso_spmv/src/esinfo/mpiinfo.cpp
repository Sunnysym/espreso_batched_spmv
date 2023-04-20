
#include "mpiinfo.h"
#include "eslog.hpp"
#include "wrappers/mpi/communication.h"

int espreso::info::mpi::rank = 0;
int espreso::info::mpi::size = 1;
MPI_Comm espreso::info::mpi::comm = MPI_COMM_WORLD;

int espreso::info::mpi::irank = 0;
int espreso::info::mpi::isize = 1;
MPI_Comm espreso::info::mpi::icomm = MPI_COMM_SELF;

int espreso::info::mpi::grank = 0;
int espreso::info::mpi::gsize = 1;
MPI_Comm espreso::info::mpi::gcomm = MPI_COMM_WORLD;

int espreso::info::mpi::threading = 0;

using namespace espreso::info;

void mpi::printStatsMPI(std::string eventName, double num) {
	Communication::barrier();
	std::vector<double> allNum(mpi::size);
	MPI_Gather(&num, 1, MPI_DOUBLE, &allNum[0], 1, MPI_DOUBLE, 0, mpi::comm);
	double g_avgNum, g_minNum, g_maxNum;
	MPI_Reduce(&num, &g_avgNum, 1, MPI_DOUBLE, MPI_SUM, 0, mpi::comm);
	g_avgNum= g_avgNum / mpi::size;

	MPI_Reduce(&num, &g_minNum, 1, MPI_DOUBLE, MPI_MIN, 0, mpi::comm);
	// int e = MPI_Allreduce(&num, &g_minNum, 1, MPI_DOUBLE, MPI_MIN, mpi::comm);
	MPI_Reduce(&num, &g_maxNum, 1, MPI_DOUBLE, MPI_MAX, 0, mpi::comm);

	// printf("event: %s, rank: %03d, num: %.5f, g_min: %.5f, e: %d\n", eventName.c_str(), mpi::rank, num, g_minNum, e);

	int name_length = 50;
	int val_length  = 12;
	std::stringstream ss;
	ss	<< std::setw(name_length) << std::left << eventName
		<< " avg.: " << std::fixed << std::setw(val_length) << g_avgNum
		<< " min.: " << std::setw(val_length) << g_minNum
		<< " max.: " << std::setw(val_length) << g_maxNum << std::endl;
	
	for (esint i = 0; i < mpi::size; i++) {
		ss << std::fixed << std::setw(3) << "R: " << std::setw(5) << i << std::setw(15) << allNum[i];
		if ((i + 1) % 4 == 0) {
			ss << std::endl;
		}
	}
	
	if (mpi::rank == 0) {
		printf("%s\n", ss.str().c_str());
	}
	
	Communication::barrier();
}

void mpi::init(int *argc, char ***argv)
{
	MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &threading);

	mpi::comm = MPI_COMM_WORLD;
	MPI_Comm_rank(mpi::comm, &mpi::rank);
	MPI_Comm_size(mpi::comm, &mpi::size);

	mpi::grank = mpi::rank;
	mpi::gsize = mpi::size;
	mpi::gcomm = mpi::comm;
}

void mpi::init(MPI_Comm comm)
{
	mpi::comm = comm;
	MPI_Comm_rank(mpi::comm, &mpi::rank);
	MPI_Comm_size(mpi::comm, &mpi::size);

	mpi::grank = mpi::rank;
	mpi::gsize = mpi::size;
	mpi::gcomm = mpi::comm;
}

bool mpi::divide(int meshDuplication)
{
	if (meshDuplication == 1) {
		return true;
	}

	if (espreso::info::mpi::size % meshDuplication != 0) {
		return false;
	}

	int color = mpi::rank / (mpi::size / meshDuplication);

	MPI_Comm_split(mpi::gcomm, color, mpi::grank, &mpi::comm);
	MPI_Comm_rank(mpi::comm, &mpi::rank);
	MPI_Comm_size(mpi::comm, &mpi::size);

	MPI_Comm_split(mpi::gcomm, mpi::rank, mpi::grank, &mpi::icomm);
	MPI_Comm_rank(mpi::icomm, &mpi::irank);
	MPI_Comm_size(mpi::icomm, &mpi::isize);

	MPITools::reinit();
	eslog::reinit();

	return true;
}

void mpi::finish()
{
	if (mpi::isize > 1) {
		MPI_Comm_free(&mpi::comm);
		MPI_Comm_free(&mpi::icomm);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}


