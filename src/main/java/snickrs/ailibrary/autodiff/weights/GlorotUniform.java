package snickrs.ailibrary.autodiff.weights;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class GlorotUniform implements WeightInitScheme {
	public GlorotUniform() {
	}
	public INDArray init(long ... dims) {
        long fanIn = dims[0];
        long fanOut = dims.length > 1 ? dims[1] : 1;
        for (int i = 2; i < dims.length; i++) {
            fanIn *= dims[i];
            fanOut *= dims[i];
        }
        float limit = (float) Math.sqrt(6.0f/(fanIn + fanOut));
		return Nd4j.rand(Nd4j.getDistributions().createUniform(-limit, limit), dims);
	}
}
