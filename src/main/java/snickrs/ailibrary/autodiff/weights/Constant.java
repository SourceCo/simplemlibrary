package snickrs.ailibrary.autodiff.weights;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Constant implements WeightInitScheme {
	public float constant;
	public Constant() {
		this.constant = 0.0f;
	}
	public Constant(float constant) {
		this.constant = constant;
	}
	public INDArray init(long ... dims) {
		return Nd4j.zeros(dims).add(constant);
	}
}
