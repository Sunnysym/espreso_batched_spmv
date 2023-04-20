
#include "itersolvercpu.h"
#include "basis/utilities/utils.h"

#include "esinfo/ecfinfo.h"
#include "mpi.h"
#include "wrappers/mpi/communication.h"



using namespace espreso;

// *** Action of K+ routines *********************************************

void helperRecive(SuperCluster& cluster)
{
	int hcnt = 0;
	for(size_t i = 0; i < myClusterInfo.commRank.size(); i++) {
		for (int d = 0; d < myClusterInfo.commDomainNum[i]; d++) {
			MPI_Irecv(&cluster.recvCompressedTmp2[hcnt][0], cluster.helpM[hcnt].rows, MPI_DOUBLE, myClusterInfo.commRank[i], 30000 + d, info::mpi::comm, &cluster.request_recv1[hcnt]);
			hcnt++;
		}
	}
	MPI_Waitall(cluster.helpSiz, &cluster.request_recv1[0], MPI_STATUSES_IGNORE);
	// Communication::barrier();
}

void helperSend(SuperCluster& cluster)
{
	// Communication::barrier();
	int hcnt = 0;
	for(size_t i = 0; i < myClusterInfo.commRank.size(); i++) {
		for (int d = 0; d < myClusterInfo.commDomainNum[i]; d++) {
			MPI_Isend(&cluster.sendCompressedTmp[hcnt][0], cluster.helpM[hcnt].rows, MPI_DOUBLE, myClusterInfo.commRank[i], 40000 + d, info::mpi::comm, &cluster.request_send2[hcnt]);
			hcnt++;
		}
	}
	MPI_Waitall(cluster.helpSiz, &cluster.request_send2[0], MPI_STATUSES_IGNORE);
}

void helpeeSend(SuperCluster& cluster)
{
	int hcnt = 0;
	for(size_t i = 0; i < myClusterInfo.commRank.size(); i++){
		for(int d = 0; d < myClusterInfo.commDomainNum[i]; d++){
			int domainId = d + myClusterInfo.commDomainBegin[i];
			MPI_Isend(&cluster.domains[domainId]->compressed_tmp2[0], cluster.domains[domainId]->B1Kplus.rows, MPI_DOUBLE, myClusterInfo.commRank[i], 30000 + d, info::mpi::comm, &cluster.request_send1[hcnt]);
			hcnt++;
		}
	}
	MPI_Waitall(cluster.helpSiz, &cluster.request_send1[0], MPI_STATUSES_IGNORE);
	// Communication::barrier();
}

void helpeeRecive(SuperCluster& cluster)
{
	// Communication::barrier();
	int hcnt = 0;
	for(size_t i = 0; i < myClusterInfo.commRank.size(); i++){
		for (int d = 0; d < myClusterInfo.commDomainNum[i]; d++) {
			int domainId = d + myClusterInfo.commDomainBegin[i];
			MPI_Irecv(&cluster.domains[domainId]->compressed_tmp[0], cluster.domains[domainId]->B1Kplus.rows, MPI_DOUBLE, myClusterInfo.commRank[i], 40000 + d, info::mpi::comm, &cluster.request_recv2[hcnt]);
			hcnt++;
		}
	}
	MPI_Waitall(cluster.helpSiz, &cluster.request_recv2[0], MPI_STATUSES_IGNORE);
}

void preHelperRecive(SuperCluster& cluster)
{
	int hcnt = 0;
	for(size_t i = 0; i < myClusterInfo.commRank.size(); i++){
		for (int d = 0; d < myClusterInfo.commDomainNum[i]; d++) {
			MPI_Irecv(&cluster.recvxPrimCluster1[hcnt][0], cluster.helpG[hcnt].rows, MPI_DOUBLE, myClusterInfo.commRank[i], 70000 + d, info::mpi::comm, &cluster.request_recv1[hcnt]);
			hcnt++;
		}
	}
	MPI_Waitall(cluster.helpSiz, &cluster.request_recv1[0], MPI_STATUSES_IGNORE);
}

void preHelperSend(SuperCluster& cluster)
{
	int hcnt = 0;
	for(size_t i = 0; i < myClusterInfo.commRank.size(); i++){
		for (int d = 0; d < myClusterInfo.commDomainNum[i]; d++) {
			MPI_Isend(&cluster.sendxPrimCluster2[hcnt][0], cluster.helpG[hcnt].rows, MPI_DOUBLE, myClusterInfo.commRank[i], 80000 + d, info::mpi::comm, &cluster.request_send2[hcnt]);
			hcnt++;
		}
	}
	MPI_Waitall(cluster.helpSiz, &cluster.request_send2[0], MPI_STATUSES_IGNORE);
}

void preHelpeeSend(SuperCluster& cluster)
{
	int hcnt = 0;
	for(size_t i = 0; i < myClusterInfo.commRank.size(); i++){
		for(int d = 0; d < myClusterInfo.commDomainNum[i]; d++){
			int domainId = d + myClusterInfo.commDomainBegin[i];
			MPI_Isend(cluster.x_prim_cluster1[domainId]->data(), cluster.domains[domainId]->Prec.rows, MPI_DOUBLE, myClusterInfo.commRank[i], 70000 + d, info::mpi::comm, &cluster.request_send1[hcnt]);
			hcnt++;
		}
	}
	MPI_Waitall(cluster.helpSiz, &cluster.request_send1[0], MPI_STATUSES_IGNORE);
}

void preHelpeeRecive(SuperCluster& cluster)
{
	int hcnt = 0;
	for(size_t i = 0; i < myClusterInfo.commRank.size(); i++){
		for (int d = 0; d < myClusterInfo.commDomainNum[i]; d++) {
			int domainId = d + myClusterInfo.commDomainBegin[i];
			MPI_Irecv(cluster.x_prim_cluster2[domainId]->data(), cluster.domains[domainId]->Prec.rows, MPI_DOUBLE, myClusterInfo.commRank[i], 80000 + d, info::mpi::comm, &cluster.request_recv2[hcnt]);
			hcnt++;
		}
	}
	MPI_Waitall(cluster.helpSiz, &cluster.request_recv2[0], MPI_STATUSES_IGNORE);
}

void IterSolverCPU::runDCU(std::vector<MVInfo>& jobs, std::vector<std::vector<int> >& threadJobs)
{
#ifdef USE_DEVICE
	int deviceCount = info::ecf->input.device_count;
	int deviceStreamCount = info::ecf->input.device_stream_count;
	#pragma omp parallel for
	for (int i = 0; i < jobs.size(); ++i)
	{
		jobs[i].B1Kplus->copyInMem(jobs[i].compressedTmp2);
	}

	if (info::ecf->input.use_device_async)
	{
		#pragma omp parallel num_threads(deviceCount)
		{
			int deviceId = omp_get_thread_num();
			hipSetDevice(deviceId);
			for (int i = 0; i < threadJobs[deviceId].size(); ++i) {
				int d = threadJobs[deviceId][i];
				int streamId = i % deviceStreamCount;
				jobs[d].B1Kplus->DenseMatVecDCUAsync(deviceStreams[deviceId][streamId], deviceStreamHandles[deviceId][streamId]);
			}
			hipDeviceSynchronize();	
		}
	} 
	else
	{
		#pragma omp parallel num_threads(deviceCount)
		{
			int deviceId = omp_get_thread_num();
			hipSetDevice(deviceId);
			for (int i = 0; i < threadJobs[deviceId].size(); ++i) {
				int d = threadJobs[deviceId][i];
				jobs[d].B1Kplus->DenseMatVecDCU(deviceHandles[deviceId]);
			}
		}
	}

	#pragma omp parallel for
	for (int i = 0; i < jobs.size(); ++i)
	{
		jobs[i].B1Kplus->copyBackMem(jobs[i].compressedTmp);
	}
#endif
}

void runCPU(std::vector<MVInfo>& jobs)
{
	#pragma omp parallel for
	for (int i = 0; i < jobs.size(); ++i)
	{
		jobs[i].B1Kplus->copyInMem(jobs[i].compressedTmp2);
	}

	#pragma omp parallel for
	for (size_t d = 0; d < jobs.size(); ++d) {
		jobs[d].B1Kplus->DenseMatVecCPU();
	}

	#pragma omp parallel for
	for (int i = 0; i < jobs.size(); ++i)
	{
		jobs[i].B1Kplus->copyBackMem(jobs[i].compressedTmp);
	}
}


void IterSolverCPU::apply_A_l_comp_dom_B( TimeEval & time_eval, SuperCluster & cluster, SEQ_VECTOR<double> & x_in, SEQ_VECTOR<double> & y_out) {

	 time_eval.totalTime.start();

    if (cluster.USE_KINV == 1 && cluster.USE_HFETI == 1) {
		time_eval.timeEvents[0].start();
		#pragma omp parallel for
		for (size_t d = 0; d < cluster.domains.size(); d++) {
			for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++) {
				cluster.domains[d]->compressed_tmp2[i] = x_in[ cluster.domains[d]->lambda_map_sub_local[i]];
			}
			cluster.domains[d]->B1_comp_dom.MatVec  (cluster.domains[d]->compressed_tmp2, *cluster.x_prim_cluster1[d], 'T');
		}
		time_eval.timeEvents[0].end();
		
		time_eval.timeEvents[1].start();
		if (info::ecf->input.use_mpi_balance)
		{
			if (myClusterInfo.isHelper) 
			{
				helperRecive(cluster);
			}
			else
			{
				helpeeSend(cluster);
			}

			if (info::ecf->input.use_device) 
			{
				runDCU(mvjobs, threadSc);
			}
			else
			{
				runCPU(mvjobs);
			}

			if (myClusterInfo.isHelper) 
			{
				helperSend(cluster);
			}
			else
			{
				helpeeRecive(cluster);
			}
		}
		else
		{
			if (info::ecf->input.use_device)
			{
				runDCU(mvjobs, threadSc);
			}
			else
			{
				#pragma omp parallel for
				for (size_t d = 0; d < cluster.domains.size(); d++) {
					cluster.domains[d]->B1Kplus.DenseMatVec (cluster.domains[d]->compressed_tmp2, cluster.domains[d]->compressed_tmp);
				}
			}
		}
		 time_eval.timeEvents[1].end();
		 time_eval.timeEvents[2].start();
        cluster.multKplusGlobal_Kinv( cluster.x_prim_cluster1 );
		 time_eval.timeEvents[2].end();

        std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);
		 time_eval.timeEvents[3].start();
        for (size_t d = 0; d < cluster.domains.size(); d++) {
            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += cluster.domains[d]->compressed_tmp[i];
        }

        SEQ_VECTOR < double > y_out_tmp;
        for (size_t d = 0; d < cluster.domains.size(); d++) {
            y_out_tmp.resize( cluster.domains[d]->B1_comp_dom.rows );
            cluster.domains[d]->B1_comp_dom.MatVec (*cluster.x_prim_cluster1[d], y_out_tmp, 'N', 0, 0, 0.0); // will add (summation per elements) all partial results into y_out

            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += y_out_tmp[i];
        }
         time_eval.timeEvents[3].end();

    }


    if (cluster.USE_KINV == 1 && cluster.USE_HFETI == 0) {
         time_eval.timeEvents[0].start();
         time_eval.timeEvents[0].end();

         time_eval.timeEvents[1].start();
        #pragma omp parallel for
        for (size_t d = 0; d < cluster.domains.size(); d++) {
            SEQ_VECTOR < double > x_in_tmp ( cluster.domains[d]->B1_comp_dom.rows );
            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
                x_in_tmp[i] = x_in[ cluster.domains[d]->lambda_map_sub_local[i]];
            cluster.domains[d]->B1Kplus.DenseMatVec ( x_in_tmp, cluster.domains[d]->compressed_tmp);
        }
         time_eval.timeEvents[1].end();

         time_eval.timeEvents[2].start();
        std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);
        for (size_t d = 0; d < cluster.domains.size(); d++) {
            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += cluster.domains[d]->compressed_tmp[i];
        }
         time_eval.timeEvents[2].end();
    }


    if (cluster.USE_KINV == 0) {
    	 time_eval.timeEvents[0].start();
 		#pragma omp parallel for
		for (size_t d = 0; d < cluster.domains.size(); d++) {
			SEQ_VECTOR < double > x_in_tmp ( cluster.domains[d]->B1_comp_dom.rows, 0.0 );
			for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
				x_in_tmp[i] = x_in[ cluster.domains[d]->lambda_map_sub_local[i]];
			cluster.domains[d]->B1_comp_dom.MatVec (x_in_tmp, *cluster.x_prim_cluster1[d], 'T');
		}
         time_eval.timeEvents[0].end();

         time_eval.timeEvents[1].start();
        if (cluster.USE_HFETI == 0) {
        	cluster.multKplusFETI (cluster.x_prim_cluster1);
		} else {
    		cluster.multKplusHFETI(cluster.x_prim_cluster1);
        }
         time_eval.timeEvents[1].end();


         time_eval.timeEvents[2].start();
		std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);

		SEQ_VECTOR < double > y_out_tmp;
		for (size_t d = 0; d < cluster.domains.size(); d++) {
			y_out_tmp.resize( cluster.domains[d]->B1_comp_dom.rows );
			cluster.domains[d]->B1_comp_dom.MatVec (*cluster.x_prim_cluster1[d], y_out_tmp, 'N', 0, 0, 0.0); // will add (summation per elements) all partial results into y_out

			for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
				cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += y_out_tmp[i];
		}

		time_eval.timeEvents[2].end();
    }

     time_eval.timeEvents[4].start();
    All_Reduce_lambdas_compB(cluster, cluster.compressed_tmp, y_out);
     time_eval.timeEvents[4].end();

     time_eval.totalTime.end();

}

void IterSolverCPU::apply_A_l_comp_dom_B_opt2( TimeEval & time_eval,  TimeEval & time_eval2, SuperCluster & cluster, SEQ_VECTOR<double> & x_in, SEQ_VECTOR<double> & y_out) {

	 time_eval.totalTime.start();

    if (cluster.USE_KINV == 1 && cluster.USE_HFETI == 1) {
		time_eval.timeEvents[0].start();
		#pragma omp parallel for
		for (size_t d = 0; d < cluster.domains.size(); d++) {
			for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++) {
				cluster.domains[d]->compressed_tmp2[i] = x_in[ cluster.domains[d]->lambda_map_sub_local[i]];
			}
			cluster.domains[d]->B1_comp_dom.MatVec  (cluster.domains[d]->compressed_tmp2, *cluster.x_prim_cluster1[d], 'T');
		}
		time_eval.timeEvents[0].end();
		
		time_eval.timeEvents[1].start();
		if (info::ecf->input.use_mpi_balance)
		{
			if (myClusterInfo.isHelper) 
			{
				helperRecive(cluster);
			}
			else
			{
				helpeeSend(cluster);
			}

			if (info::ecf->input.use_device) 
			{
				runDCU(mvjobs, threadSc);
			}
			else
			{
				runCPU(mvjobs);
			}

			if (myClusterInfo.isHelper) 
			{
				helperSend(cluster);
			}
			else
			{
				helpeeRecive(cluster);
			}
		}
		else
		{
			if (info::ecf->input.use_device)
			{
				runDCU(mvjobs, threadSc);
			}
			else
			{
				#pragma omp parallel for
				for (size_t d = 0; d < cluster.domains.size(); d++) {
					cluster.domains[d]->B1Kplus.DenseMatVec (cluster.domains[d]->compressed_tmp2, cluster.domains[d]->compressed_tmp);
				}
			}
		}
		 time_eval.timeEvents[1].end();
		 time_eval.timeEvents[2].start();
        cluster.multKplusGlobal_Kinv_opt2(time_eval2, cluster.x_prim_cluster1 );
		 time_eval.timeEvents[2].end();

        std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);
		 time_eval.timeEvents[3].start();
        for (size_t d = 0; d < cluster.domains.size(); d++) {
            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += cluster.domains[d]->compressed_tmp[i];
        }

        SEQ_VECTOR < double > y_out_tmp;
        for (size_t d = 0; d < cluster.domains.size(); d++) {
            y_out_tmp.resize( cluster.domains[d]->B1_comp_dom.rows );
            cluster.domains[d]->B1_comp_dom.MatVec (*cluster.x_prim_cluster1[d], y_out_tmp, 'N', 0, 0, 0.0); // will add (summation per elements) all partial results into y_out

            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += y_out_tmp[i];
        }
         time_eval.timeEvents[3].end();

    }


    if (cluster.USE_KINV == 1 && cluster.USE_HFETI == 0) {
         time_eval.timeEvents[0].start();
         time_eval.timeEvents[0].end();

         time_eval.timeEvents[1].start();
        #pragma omp parallel for
        for (size_t d = 0; d < cluster.domains.size(); d++) {
            SEQ_VECTOR < double > x_in_tmp ( cluster.domains[d]->B1_comp_dom.rows );
            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
                x_in_tmp[i] = x_in[ cluster.domains[d]->lambda_map_sub_local[i]];
            cluster.domains[d]->B1Kplus.DenseMatVec ( x_in_tmp, cluster.domains[d]->compressed_tmp);
        }
         time_eval.timeEvents[1].end();

         time_eval.timeEvents[2].start();
        std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);
        for (size_t d = 0; d < cluster.domains.size(); d++) {
            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += cluster.domains[d]->compressed_tmp[i];
        }
         time_eval.timeEvents[2].end();
    }


    if (cluster.USE_KINV == 0) {
    	 time_eval.timeEvents[0].start();
 		#pragma omp parallel for
		for (size_t d = 0; d < cluster.domains.size(); d++) {
			SEQ_VECTOR < double > x_in_tmp ( cluster.domains[d]->B1_comp_dom.rows, 0.0 );
			for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
				x_in_tmp[i] = x_in[ cluster.domains[d]->lambda_map_sub_local[i]];
			cluster.domains[d]->B1_comp_dom.MatVec (x_in_tmp, *cluster.x_prim_cluster1[d], 'T');
		}
         time_eval.timeEvents[0].end();

         time_eval.timeEvents[1].start();
        if (cluster.USE_HFETI == 0) {
        	cluster.multKplusFETI (cluster.x_prim_cluster1);
		} else {
    		cluster.multKplusHFETI(cluster.x_prim_cluster1);
        }
         time_eval.timeEvents[1].end();


         time_eval.timeEvents[2].start();
		std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);

		SEQ_VECTOR < double > y_out_tmp;
		for (size_t d = 0; d < cluster.domains.size(); d++) {
			y_out_tmp.resize( cluster.domains[d]->B1_comp_dom.rows );
			cluster.domains[d]->B1_comp_dom.MatVec (*cluster.x_prim_cluster1[d], y_out_tmp, 'N', 0, 0, 0.0); // will add (summation per elements) all partial results into y_out

			for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
				cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += y_out_tmp[i];
		}

		time_eval.timeEvents[2].end();
    }

     time_eval.timeEvents[4].start();
    All_Reduce_lambdas_compB(cluster, cluster.compressed_tmp, y_out);
     time_eval.timeEvents[4].end();

     time_eval.totalTime.end();

}


void IterSolverCPU::apply_A_l_comp_dom_B_P( TimeEval & time_eval, SuperCluster & cluster, SEQ_VECTOR<double> & x_in, SEQ_VECTOR<double> & y_out) {

	 time_eval.totalTime.start();

//    if (cluster.USE_KINV == 1 && cluster.USE_HFETI == 1) {
//         time_eval.timeEvents[0].start();
//        #pragma omp parallel for
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++) {
//                cluster.domains[d]->compressed_tmp2[i] = x_in[ cluster.domains[d]->lambda_map_sub_local[i]];
//            }
//            cluster.domains[d]->B1Kplus.DenseMatVec (cluster.domains[d]->compressed_tmp2, cluster.domains[d]->compressed_tmp);
//            cluster.domains[d]->B1_comp_dom.MatVec  (cluster.domains[d]->compressed_tmp2, *cluster.x_prim_cluster1[d], 'T');
//        }
//         time_eval.timeEvents[0].end();
//
//         time_eval.timeEvents[1].start();
//        cluster.multKplusGlobal_Kinv( cluster.x_prim_cluster1 );
//         time_eval.timeEvents[1].end();
//
//         time_eval.timeEvents[2].start();
//        std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
//                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += cluster.domains[d]->compressed_tmp[i];
//        }
//
//        SEQ_VECTOR < double > y_out_tmp;
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            y_out_tmp.resize( cluster.domains[d]->B1_comp_dom.rows );
//            cluster.domains[d]->B1_comp_dom.MatVec (*cluster.x_prim_cluster1[d], y_out_tmp, 'N', 0, 0, 0.0); // will add (summation per elements) all partial results into y_out
//
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
//                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += y_out_tmp[i];
//        }
//         time_eval.timeEvents[2].end();
//
//    }
//
//
//    if (cluster.USE_KINV == 1 && cluster.USE_HFETI == 0) {
//         time_eval.timeEvents[0].start();
//         time_eval.timeEvents[0].end();
//
//         time_eval.timeEvents[1].start();
//        #pragma omp parallel for
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            SEQ_VECTOR < double > x_in_tmp ( cluster.domains[d]->B1_comp_dom.rows );
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
//                x_in_tmp[i] = x_in[ cluster.domains[d]->lambda_map_sub_local[i]];
//            cluster.domains[d]->B1Kplus.DenseMatVec ( x_in_tmp, cluster.domains[d]->compressed_tmp);
//        }
//         time_eval.timeEvents[1].end();
//
//         time_eval.timeEvents[2].start();
//        std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
//                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += cluster.domains[d]->compressed_tmp[i];
//        }
//         time_eval.timeEvents[2].end();
//    }


    if (cluster.USE_KINV == 0) {
    	 time_eval.timeEvents[0].start();
 		#pragma omp parallel for
		for (size_t d = 0; d < cluster.domains.size(); d++) {
			SEQ_VECTOR < double > x_in_tmp ( cluster.domains[d]->B1_comp_dom.rows, 0.0 );
			for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub.size(); i++)
				x_in_tmp[i] = x_in[ cluster.domains[d]->lambda_map_sub[i]];
			cluster.domains[d]->B1_comp_dom.MatVec (x_in_tmp, *cluster.x_prim_cluster1[d], 'T');
		}
         time_eval.timeEvents[0].end();

         time_eval.timeEvents[1].start();
        if (cluster.USE_HFETI == 0) {
        	cluster.multKplusFETI (cluster.x_prim_cluster1);
		} else {
    		cluster.multKplusHFETI(cluster.x_prim_cluster1);
        }
         time_eval.timeEvents[1].end();


         time_eval.timeEvents[2].start();
		std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);

		SEQ_VECTOR < double > y_out_tmp;
		for (size_t d = 0; d < cluster.domains.size(); d++) {
			y_out_tmp.resize( cluster.domains[d]->B1_comp_dom.rows );
			cluster.domains[d]->B1_comp_dom.MatVec (*cluster.x_prim_cluster1[d], y_out_tmp, 'N', 0, 0, 0.0); // will add (summation per elements) all partial results into y_out

			for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
				cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += y_out_tmp[i];
		}

		time_eval.timeEvents[2].end();
    }

     time_eval.timeEvents[3].start();
    All_Reduce_lambdas_compB(cluster, cluster.compressed_tmp, y_out);
     time_eval.timeEvents[3].end();

 	for (size_t i = 0; i < cluster.my_lamdas_indices.size(); i++)
 		y_out[i] = y_out[i] * cluster.my_lamdas_ddot_filter[i];

	SEQ_VECTOR < double > y_out_tmp (x_in.size(), 0.0);
	for (size_t d = 0; d < cluster.domains.size(); d++) {
		for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++) {

			y_out_tmp [ cluster.domains[d]->lambda_map_sub[i] ]   +=  y_out [ cluster.domains[d]->lambda_map_sub_local[i] ];
																	  y_out [ cluster.domains[d]->lambda_map_sub_local[i] ] = 0.0;

			//cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += y_out_tmp[i];

		}
	}

	y_out = y_out_tmp;

     time_eval.totalTime.end();

}


void IterSolverCPU::apply_A_l_comp_dom_B_P_local( TimeEval & time_eval, SuperCluster & cluster, SEQ_VECTOR<double> & x_in, SEQ_VECTOR<double> & y_out) {

//	 time_eval.totalTime.start();

    if (cluster.USE_KINV == 1 && cluster.USE_HFETI == 1) {

    	eslog::error("Hybrid FETI not supported in apply_A_l_comp_dom_B_P_local.\n");

//         time_eval.timeEvents[0].start();
//        #pragma omp parallel for
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++) {
//                cluster.domains[d]->compressed_tmp2[i] = x_in[ cluster.domains[d]->lambda_map_sub_local[i]];
//            }
//            cluster.domains[d]->B1Kplus.DenseMatVec (cluster.domains[d]->compressed_tmp2, cluster.domains[d]->compressed_tmp);
//            cluster.domains[d]->B1_comp_dom.MatVec  (cluster.domains[d]->compressed_tmp2, *cluster.x_prim_cluster1[d], 'T');
//        }
//         time_eval.timeEvents[0].end();
//
//         time_eval.timeEvents[1].start();
//        cluster.multKplusGlobal_Kinv( cluster.x_prim_cluster1 );
//         time_eval.timeEvents[1].end();
//
//         time_eval.timeEvents[2].start();
//        std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
//                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += cluster.domains[d]->compressed_tmp[i];
//        }
//
//        SEQ_VECTOR < double > y_out_tmp;
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            y_out_tmp.resize( cluster.domains[d]->B1_comp_dom.rows );
//            cluster.domains[d]->B1_comp_dom.MatVec (*cluster.x_prim_cluster1[d], y_out_tmp, 'N', 0, 0, 0.0); // will add (summation per elements) all partial results into y_out
//
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
//                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += y_out_tmp[i];
//        }
//         time_eval.timeEvents[2].end();
//
    }
//
//
    if (cluster.USE_KINV == 1 && cluster.USE_HFETI == 0) {

    	eslog::error("Local Schur Complement (LSC) method not supported in apply_A_l_comp_dom_B_P_local.\n");

//         time_eval.timeEvents[0].start();
//         time_eval.timeEvents[0].end();
//
//         time_eval.timeEvents[1].start();
//        #pragma omp parallel for
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            SEQ_VECTOR < double > x_in_tmp ( cluster.domains[d]->B1_comp_dom.rows );
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
//                x_in_tmp[i] = x_in[ cluster.domains[d]->lambda_map_sub_local[i]];
//            cluster.domains[d]->B1Kplus.DenseMatVec ( x_in_tmp, cluster.domains[d]->compressed_tmp);
//        }
//         time_eval.timeEvents[1].end();
//
//         time_eval.timeEvents[2].start();
//        std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
//                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += cluster.domains[d]->compressed_tmp[i];
//        }
//         time_eval.timeEvents[2].end();
    }


    if (cluster.USE_KINV == 0) {
//    	 time_eval.timeEvents[0].start();
 		#pragma omp parallel for
		for (size_t d = 0; d < cluster.domains.size(); d++) {
			SEQ_VECTOR < double > x_in_tmp ( cluster.domains[d]->B1_comp_dom.rows, 0.0 );
			for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub.size(); i++)
				x_in_tmp[i] = x_in[ cluster.domains[d]->lambda_map_sub[i]];
			cluster.domains[d]->B1_comp_dom.MatVec (x_in_tmp, *cluster.x_prim_cluster1[d], 'T');
		}
//         time_eval.timeEvents[0].end();

//         time_eval.timeEvents[1].start();
        if (cluster.USE_HFETI == 0) {
        	cluster.multKplusFETI (cluster.x_prim_cluster1);
		} else {
    		cluster.multKplusHFETI(cluster.x_prim_cluster1);
        }
//         time_eval.timeEvents[1].end();


//         time_eval.timeEvents[2].start();
		std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);

		SEQ_VECTOR < double > y_out_tmp;
		for (size_t d = 0; d < cluster.domains.size(); d++) {
			y_out_tmp.resize( cluster.domains[d]->B1_comp_dom.rows );
			cluster.domains[d]->B1_comp_dom.MatVec (*cluster.x_prim_cluster1[d], y_out_tmp, 'N', 0, 0, 0.0); // will add (summation per elements) all partial results into y_out

			for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
				cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += y_out_tmp[i];
		}

//		time_eval.timeEvents[2].end();
    }

//     time_eval.timeEvents[3].start();
    //All_Reduce_lambdas_compB(cluster, cluster.compressed_tmp, y_out);
     y_out = cluster.compressed_tmp;
//     time_eval.timeEvents[3].end();

// 	for (size_t i = 0; i < cluster.my_lamdas_indices.size(); i++)
//		y_out[i] = y_out[i] * cluster.my_lamdas_ddot_filter[i];

	SEQ_VECTOR < double > y_out_tmp (x_in.size(), 0.0);
	for (size_t d = 0; d < cluster.domains.size(); d++) {
		for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++) {

			y_out_tmp [ cluster.domains[d]->lambda_map_sub[i] ]   +=  y_out [ cluster.domains[d]->lambda_map_sub_local[i] ];
			y_out [ cluster.domains[d]->lambda_map_sub_local[i] ] = 0.0;

		}
	}

	y_out = y_out_tmp;

//     time_eval.totalTime.end();

}

void IterSolverCPU::apply_A_l_comp_dom_B_P_local_sparse( TimeEval & time_eval, SuperCluster & cluster, SEQ_VECTOR<esint> & in_indices, SEQ_VECTOR<double> & in_values, SEQ_VECTOR<esint> & out_indices, SEQ_VECTOR<double> & out_values) {

//	 time_eval.totalTime.start();

    if (cluster.USE_KINV == 1 && cluster.USE_HFETI == 1) {

    	eslog::error("Hybrid FETI not supported in apply_A_l_comp_dom_B_P_local_sparse.\n");

//         time_eval.timeEvents[0].start();
//        #pragma omp parallel for
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++) {
//                cluster.domains[d]->compressed_tmp2[i] = x_in[ cluster.domains[d]->lambda_map_sub_local[i]];
//            }
//            cluster.domains[d]->B1Kplus.DenseMatVec (cluster.domains[d]->compressed_tmp2, cluster.domains[d]->compressed_tmp);
//            cluster.domains[d]->B1_comp_dom.MatVec  (cluster.domains[d]->compressed_tmp2, *cluster.x_prim_cluster1[d], 'T');
//        }
//         time_eval.timeEvents[0].end();
//
//         time_eval.timeEvents[1].start();
//        cluster.multKplusGlobal_Kinv( cluster.x_prim_cluster1 );
//         time_eval.timeEvents[1].end();
//
//         time_eval.timeEvents[2].start();
//        std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
//                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += cluster.domains[d]->compressed_tmp[i];
//        }
//
//        SEQ_VECTOR < double > y_out_tmp;
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            y_out_tmp.resize( cluster.domains[d]->B1_comp_dom.rows );
//            cluster.domains[d]->B1_comp_dom.MatVec (*cluster.x_prim_cluster1[d], y_out_tmp, 'N', 0, 0, 0.0); // will add (summation per elements) all partial results into y_out
//
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
//                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += y_out_tmp[i];
//        }
//         time_eval.timeEvents[2].end();
//
    }


    if (cluster.USE_KINV == 1 && cluster.USE_HFETI == 0) {

    	eslog::error("Local Schur Complement (LSC) method not supported in apply_A_l_comp_dom_B_P_local_sparse.\n");

//         time_eval.timeEvents[0].start();
//         time_eval.timeEvents[0].end();
//
//         time_eval.timeEvents[1].start();
//        #pragma omp parallel for
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            SEQ_VECTOR < double > x_in_tmp ( cluster.domains[d]->B1_comp_dom.rows );
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
//                x_in_tmp[i] = x_in[ cluster.domains[d]->lambda_map_sub_local[i]];
//            cluster.domains[d]->B1Kplus.DenseMatVec ( x_in_tmp, cluster.domains[d]->compressed_tmp);
//        }
//         time_eval.timeEvents[1].end();
//
//         time_eval.timeEvents[2].start();
//        std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);
//        for (size_t d = 0; d < cluster.domains.size(); d++) {
//            for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
//                cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += cluster.domains[d]->compressed_tmp[i];
//        }
//         time_eval.timeEvents[2].end();
    }


    if (cluster.USE_KINV == 0) {
//    	 time_eval.timeEvents[0].start();
    	SEQ_VECTOR <int> is_empty ( cluster.x_prim_cluster1.size(), true);
 		#pragma omp parallel for
		for (size_t d = 0; d < cluster.domains.size(); d++) {
			SEQ_VECTOR < double > x_in_tmp ( cluster.domains[d]->B1_comp_dom.rows, 0.0 );

//			for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub.size(); i++)
//				x_in_tmp[i] = x_in[ cluster.domains[d]->lambda_map_sub[i]];

			esint iter_in = 0;
			esint iter_lm = 0;

			do {

				if (cluster.domains[d]->lambda_map_sub[iter_lm] == in_indices[iter_in]) {
					x_in_tmp[iter_lm] = in_values[iter_in];
					iter_in++;
					iter_lm++;
					is_empty[d] = false;
				} else {
					if (cluster.domains[d]->lambda_map_sub[iter_lm] > in_indices[iter_in]) {
						iter_in++;
					} else {
					//if (cluster.domains[d]->lambda_map_sub[iter_lm] < in_indices[iter_in])
						iter_lm++;
					}
				}

			} while (iter_lm != (esint)cluster.domains[d]->lambda_map_sub.size() && iter_in != (esint)in_indices.size() );

			if (!is_empty[d])
				cluster.domains[d]->B1_comp_dom.MatVec (x_in_tmp, *cluster.x_prim_cluster1[d], 'T');
		}

//         time_eval.timeEvents[0].end();

//         time_eval.timeEvents[1].start();
        if (cluster.USE_HFETI == 0) {
        	//cluster.multKplusFETI (cluster.x_prim_cluster1);

			#pragma omp parallel for
			for (size_t d = 0; d < cluster.domains.size(); d++) {
				if (!is_empty[d])
					cluster.domains[d]->multKplusLocal(*cluster.x_prim_cluster1[d]);
			}

		} else {
    		cluster.multKplusHFETI(cluster.x_prim_cluster1);
        }
//         time_eval.timeEvents[1].end();


//         time_eval.timeEvents[2].start();
		std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);

		SEQ_VECTOR < double > y_out_tmp;
		for (size_t d = 0; d < cluster.domains.size(); d++) {
			if (!is_empty[d]) {
				y_out_tmp.resize( cluster.domains[d]->B1_comp_dom.rows );
				cluster.domains[d]->B1_comp_dom.MatVec (*cluster.x_prim_cluster1[d], y_out_tmp, 'N', 0, 0, 0.0); // will add (summation per elements) all partial results into y_out

				for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
					cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += y_out_tmp[i];
			}
		}

//		time_eval.timeEvents[2].end();
    }

	out_indices = cluster.my_lamdas_indices;
	out_values  = cluster.compressed_tmp;

//     time_eval.totalTime.end();

}



void IterSolverCPU::Apply_Prec( TimeEval & time_eval, SuperCluster & cluster, SEQ_VECTOR<double> & x_in, SEQ_VECTOR<double> & y_out ) {

	 time_eval.totalTime.start();


	if (USE_PREC == FETIConfiguration::PRECONDITIONER::DIRICHLET)
	{
		time_eval.timeEvents[0].start();
		#pragma omp parallel for
		for (size_t d = 0; d < cluster.domains.size(); d++) {
			SEQ_VECTOR < double > x_in_tmp ( cluster.domains[d]->B1_comp_dom.rows, 0.0 );
			for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
				x_in_tmp[i] = x_in[ cluster.domains[d]->lambda_map_sub_local[i]] * cluster.domains[d]->B1_scale_vec[i]; // includes B1 scaling
			
			cluster.domains[d]->B1t_DirPr.MatVec (x_in_tmp, *cluster.x_prim_cluster1[d], 'N');
		}
		time_eval.timeEvents[0].end();

		time_eval.timeEvents[1].start();
		if (info::ecf->input.use_mpi_balance)
		{
			if (myClusterInfo.isHelper) 
			{
				preHelperRecive(cluster);
			}
			else
			{
				preHelpeeSend(cluster);
			}

			if (info::ecf->input.use_device) 
			{
				runDCU(premvjobs, threadPre);
			}
			else
			{
				runCPU(premvjobs);
			}

			if (myClusterInfo.isHelper) 
			{
				preHelperSend(cluster);
			}
			else
			{
				preHelpeeRecive(cluster);
			}
		}
		else
		{
			if (info::ecf->input.use_device) 
			{
				runDCU(premvjobs, threadPre);
			}
			else
			{
				#pragma omp parallel for
				for (size_t d = 0; d < cluster.domains.size(); d++)
				{
					cluster.domains[d]->Prec.DenseMatVec(*cluster.x_prim_cluster1[d], *cluster.x_prim_cluster2[d],'N');
				}
			}
		}
		time_eval.timeEvents[1].end();
	}
	else
	{
	#pragma omp parallel for
	for (size_t d = 0; d < cluster.domains.size(); d++) {
		SEQ_VECTOR < double > x_in_tmp ( cluster.domains[d]->B1_comp_dom.rows, 0.0 );
		for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
			x_in_tmp[i] = x_in[ cluster.domains[d]->lambda_map_sub_local[i]] * cluster.domains[d]->B1_scale_vec[i]; // includes B1 scaling

		switch (USE_PREC) {
		case FETIConfiguration::PRECONDITIONER::LUMPED:
			cluster.domains[d]->B1_comp_dom.MatVec (x_in_tmp, *cluster.x_prim_cluster1[d], 'T');
			cluster.domains[d]->K.MatVec(*cluster.x_prim_cluster1[d], *cluster.x_prim_cluster2[d],'N');
			if (cluster.domains[d]->_RegMat.nnz > 0) {
				cluster.domains[d]->_RegMat.MatVecCOO(*cluster.x_prim_cluster1[d], *cluster.x_prim_cluster2[d],'N', 1.0, -1.0);
			}
			break;
		case FETIConfiguration::PRECONDITIONER::WEIGHT_FUNCTION:
			cluster.domains[d]->B1_comp_dom.MatVec (x_in_tmp, *cluster.x_prim_cluster2[d], 'T');
			break;
		case FETIConfiguration::PRECONDITIONER::DIRICHLET:
			cluster.domains[d]->B1t_DirPr.MatVec (x_in_tmp, *cluster.x_prim_cluster1[d], 'N');
			cluster.domains[d]->Prec.DenseMatVec(*cluster.x_prim_cluster1[d], *cluster.x_prim_cluster2[d],'N');
			break;
		case FETIConfiguration::PRECONDITIONER::SUPER_DIRICHLET:
			cluster.domains[d]->B1t_DirPr.MatVec (x_in_tmp, *cluster.x_prim_cluster1[d], 'N');
			cluster.domains[d]->Prec.MatVec(*cluster.x_prim_cluster1[d], *cluster.x_prim_cluster2[d],'N');
			break;
		case FETIConfiguration::PRECONDITIONER::MAGIC:
			cluster.domains[d]->B1_comp_dom.MatVec (x_in_tmp, *cluster.x_prim_cluster1[d], 'T');
			cluster.domains[d]->Prec.MatVec(*cluster.x_prim_cluster1[d], *cluster.x_prim_cluster2[d],'N');
			break;
		case FETIConfiguration::PRECONDITIONER::NONE:
			break;
		default:
			eslog::error("Not implemented preconditioner.\n");
		}

	}
	}

	std::fill( cluster.compressed_tmp.begin(), cluster.compressed_tmp.end(), 0.0);
	SEQ_VECTOR < double > y_out_tmp;
	time_eval.timeEvents[2].start();
	for (size_t d = 0; d < cluster.domains.size(); d++) {
		y_out_tmp.resize( cluster.domains[d]->B1_comp_dom.rows );


		switch (USE_PREC) {
		case FETIConfiguration::PRECONDITIONER::LUMPED:
		case FETIConfiguration::PRECONDITIONER::WEIGHT_FUNCTION:
		case FETIConfiguration::PRECONDITIONER::MAGIC:
			cluster.domains[d]->B1_comp_dom.MatVec (*cluster.x_prim_cluster2[d], y_out_tmp, 'N', 0, 0, 0.0); // will add (summation per elements) all partial results into y_out
			break;
		//TODO  check if MatVec is correct (DenseMatVec!!!)
		case FETIConfiguration::PRECONDITIONER::DIRICHLET:
			cluster.domains[d]->B1t_DirPr.MatVec (*cluster.x_prim_cluster2[d], y_out_tmp, 'T', 0, 0, 0.0); // will add (summation per elements) all partial results into y_out
			break;
		case FETIConfiguration::PRECONDITIONER::SUPER_DIRICHLET:
			cluster.domains[d]->B1t_DirPr.MatVec (*cluster.x_prim_cluster2[d], y_out_tmp, 'T', 0, 0, 0.0); // will add (summation per elements) all partial results into y_out
			break;
		case FETIConfiguration::PRECONDITIONER::NONE:
			break;
		default:
			eslog::error("Not implemented preconditioner.\n");
		}


		for (size_t i = 0; i < cluster.domains[d]->lambda_map_sub_local.size(); i++)
			cluster.compressed_tmp[ cluster.domains[d]->lambda_map_sub_local[i] ] += y_out_tmp[i] * cluster.domains[d]->B1_scale_vec[i]; // includes B1 scaling
	}
	time_eval.timeEvents[2].end();


	 time_eval.timeEvents[3].start();
	All_Reduce_lambdas_compB(cluster, cluster.compressed_tmp, y_out);
	 time_eval.timeEvents[3].end();


	 time_eval.totalTime.end();

}



