package snickrs.ailibrary.autodiff.weights;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class VarianceScaling implements WeightInitScheme {
	public float scale; // default 1.0
	public String mode; // "fan_in", "fan_out", or "fan_avg"
	public String distribution; // "truncated_normal", "untruncated_normal", or "uniform"
	public VarianceScaling() {
		this.scale = 1.0f;
		this.mode = "fan_in";
		this.distribution = "truncated_normal";
	}
	public VarianceScaling(float scale, String mode, String distribution) {
		this.scale = scale;
		this.mode = mode;
		this.distribution = distribution;
	}
	public INDArray init(long ... dims) {
        long fanIn = dims[0];
        long fanOut = dims.length > 1 ? dims[1] : 1;
        for (int i = 2; i < dims.length; i++) {
            fanIn *= dims[i];
            fanOut *= dims[i];
        }
        float n;
        switch(mode) {
        	case "fan_in":
        		n = fanIn;
        		break;
        	case "fan_out":
        		n = fanOut;
        		break;
        	case "fan_avg":
        		n = (fanIn+fanOut)/2.0f;
        		break;
        	default:
        		throw new IllegalArgumentException("Unknown mode: " + mode);
        }
        switch(distribution) {
        	case "truncated_normal":
        		return Nd4j.rand(Nd4j.getDistributions().createTruncatedNormal(0.0f, Math.sqrt(scale/n)), dims);
        	case "untruncated_normal":
        		return Nd4j.rand(Nd4j.getDistributions().createNormal(0.0f, Math.sqrt(scale/n)), dims);
        	case "uniform":
        		float limit = (float) Math.sqrt(3.0f*scale/n);
        		return Nd4j.rand(Nd4j.getDistributions().createUniform(-limit, limit), dims);
        	default:
        		throw new IllegalArgumentException("Unknown distribution: " + distribution);
        }
	}
}
