package snickrs.ailibrary.autodiff.weights;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class HeUniform implements WeightInitScheme {
	public HeUniform() {
	}
	public INDArray init(long ... dims) {
        long fanIn = dims[0];
        for (int i = 2; i < dims.length; i++) {
            fanIn *= dims[i];
        }
        float limit = (float) Math.sqrt(6.0f/fanIn);
		return Nd4j.rand(Nd4j.getDistributions().createUniform(-limit, limit), dims);
	}
}
