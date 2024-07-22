package snickrs.ailibrary.autodiff.weights;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface WeightInitScheme {
	public INDArray init(long ... dims);
}
