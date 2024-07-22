package snickrs.ailibrary.autodiff.weights;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;

public class RandomUniform implements WeightInitScheme {
	public float minval;
	public float maxval;
	public Distribution dist;
	public RandomUniform() {
		this.minval = -0.05f;
		this.maxval = 0.05f;
		this.dist = Nd4j.getDistributions().createUniform(this.minval, this.maxval);
	}
	public RandomUniform(float minval, float maxval) {
		this.minval = minval;
		this.maxval = maxval;
		this.dist = Nd4j.getDistributions().createUniform(this.minval, this.maxval);
	}
	public INDArray init(long ... dims) {
		return Nd4j.rand(dist, dims);
	}
}
