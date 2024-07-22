package snickrs.ailibrary.autodiff.weights;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LecunUniform implements WeightInitScheme {
	public LecunUniform() {
	}
	public INDArray init(long ... dims) {
        long fanIn = dims[0];
        for (int i = 2; i < dims.length; i++) {
            fanIn *= dims[i];
        }
        float limit = (float) Math.sqrt(3.0f/fanIn);
		return Nd4j.rand(Nd4j.getDistributions().createUniform(-limit, limit), dims);
	}
}
