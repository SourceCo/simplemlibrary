package snickrs.ailibrary.autodiff.weights;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Ones implements WeightInitScheme {
	public Ones() {
	}
	public INDArray init(long ... dims) {
		return Nd4j.ones(dims);
	}
}
