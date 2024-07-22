package snickrs.ailibrary.autodiff.weights;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Zeros implements WeightInitScheme {
	public Zeros() {
	}
	public INDArray init(long ... dims) {
		return Nd4j.zeros(dims);
	}
}
