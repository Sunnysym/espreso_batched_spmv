
#ifndef SRC_INPUT_MESHGENERATOR_ELEMENTS_3D_PYRAMID13_H_
#define SRC_INPUT_MESHGENERATOR_ELEMENTS_3D_PYRAMID13_H_

#include "input/parsers/meshgenerator/elements/element.h"

namespace espreso {

struct Pyramid13Generator: public ElementGenerator {

	Pyramid13Generator();

	void pushElements(std::vector<esint> &elements, const std::vector<esint> &indices) const;
	void pushNodes(std::vector<esint> &nodes, const std::vector<esint> &indices, CubeEdge edge) const;
	void pushNodes(std::vector<esint> &nodes, const std::vector<esint> &indices, CubeFace face) const;
	void pushEdge(std::vector<esint> &elements, std::vector<esint> &esize, std::vector<int> &etype, const std::vector<esint> &indices, CubeEdge edge) const;
	void pushFace(std::vector<esint> &elements, std::vector<esint> &esize, std::vector<int> &etype, const std::vector<esint> &indices, CubeFace face) const;
};

}


#endif /* SRC_INPUT_MESHGENERATOR_ELEMENTS_3D_PYRAMID13_H_ */
