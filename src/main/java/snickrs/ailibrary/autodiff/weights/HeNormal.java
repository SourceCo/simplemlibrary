package snickrs.ailibrary.autodiff.weights;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class HeNormal implements WeightInitScheme {
	public HeNormal() {
	}
	public INDArray init(long ... dims) {
        long fanIn = dims[0];
        for (int i = 2; i < dims.length; i++) {
            fanIn *= dims[i];
        }
		return Nd4j.rand(Nd4j.getDistributions().createTruncatedNormal(0.0f, Math.sqrt(2.0f/fanIn)), dims);
	}
}
