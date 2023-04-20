
#ifndef SRC_PHYSICS_KERNELS_BASEFUNCTIONS_LINE_LINE2_H_
#define SRC_PHYSICS_KERNELS_BASEFUNCTIONS_LINE_LINE2_H_

#include "physics/kernels/basefunctions/basefunctions.h"

namespace espreso {

struct Line2: public Element {

	static void setBaseFunctions(Element &self);
	void setGaussPointsForOrder(int order);
};

}

#endif /* SRC_PHYSICS_KERNELS_BASEFUNCTIONS_LINE_LINE2_H_ */
