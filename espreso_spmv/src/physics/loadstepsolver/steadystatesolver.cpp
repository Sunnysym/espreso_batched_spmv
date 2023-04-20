
#include "steadystatesolver.h"
#include "esinfo/stepinfo.h"
#include "esinfo/eslog.hpp"

#include "physics/system/linearsystem.h"
#include "physics/system/builder/builder.h"
#include "physics/substepsolver/substepsolver.h"
#include "config/ecf/physics/physicssolver/loadstep.h"

using namespace espreso;

SteadyStateSolver::SteadyStateSolver(LinearSystem *system, SubStepSolver *subStepSolver, double duration)
: LoadStepSolver(system, subStepSolver, duration)
{

}

void SteadyStateSolver::init(LoadStepSolver *previous)
{

}

void SteadyStateSolver::updateStructuralMatrices()
{
	_system->builder->matrices &= Builder::Request::K | Builder::Request::RBCf;
	_system->assemble();
}

void SteadyStateSolver::runNextSubstep()
{
	step::time.current = step::time.final;
	step::time.shift = step::time.final - step::time.current;
	_system->nextSubstep();

	_system->builder->internalForceReduction = 1;
	_system->builder->timeIntegrationConstantK = 1;
	_system->builder->timeIntegrationConstantC = 0;
	_system->builder->timeIntegrationConstantM = 0;

	eslog::solver("\n = ================================== STEADY STATE SOLVER ================================== =\n");
	eslog::solver(" =  LOAD STEP %2d, SUBSTEP %4d, TIME %10.6f, TIME STEP %10.6f, FINAL TIME %10.6f =\n", step::step.loadstep + 1, step::step.substep + 1, step::time.current, step::time.shift, step::time.final);
	eslog::solver(" = ----------------------------------------------------------------------------------------- =\n");

	_subStepSolver->solve(*this);
	_system->processSolution();

	eslog::solver(" = ========================================================================================= =\n");
	eslog::solver(" = ================================================================= run time %12.3f s =\n\n", eslog::duration());
}

