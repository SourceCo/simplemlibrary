package snickrs.ailibrary.autodiff.weights;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class GlorotNormal implements WeightInitScheme {
	public GlorotNormal() {
	}
	public INDArray init(long ... dims) {
        long fanIn = dims[0];
        long fanOut = dims.length > 1 ? dims[1] : 1;
        for (int i = 2; i < dims.length; i++) {
            fanIn *= dims[i];
            fanOut *= dims[i];
        }
		return Nd4j.rand(Nd4j.getDistributions().createTruncatedNormal(0.0f, Math.sqrt(2.0f/(fanIn + fanOut))), dims);
	}
}
