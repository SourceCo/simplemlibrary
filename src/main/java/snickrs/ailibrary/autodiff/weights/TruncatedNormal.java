package snickrs.ailibrary.autodiff.weights;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;

public class TruncatedNormal implements WeightInitScheme {
	public float mean;
	public float stdev;
	public Distribution dist;
	public TruncatedNormal() {
		this.mean = 0.0f;
		this.stdev = 0.05f;
		this.dist = Nd4j.getDistributions().createTruncatedNormal(this.mean, this.stdev);
	}
	public TruncatedNormal(float mean, float stdev) {
		this.mean = mean;
		this.stdev = stdev;
		this.dist = Nd4j.getDistributions().createTruncatedNormal(this.mean, this.stdev);
	}
	public INDArray init(long ... dims) {
		return Nd4j.rand(dist, dims);
	}
}
