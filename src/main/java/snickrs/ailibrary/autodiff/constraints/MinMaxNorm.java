package snickrs.ailibrary.autodiff.constraints;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MinMaxNorm implements Constraint {
	public float min_value;
	public float max_value;
	public float rate;
	public int axis;
	public MinMaxNorm() {
		this.min_value = 0.0f;
		this.max_value = 1.0f;
		this.rate = 1.0f;
		this.axis = 0;
	}
	public MinMaxNorm(float min_value, float max_value) {
		this.min_value = min_value;
		this.max_value = max_value;
		this.axis = 0;
	}
	public MinMaxNorm(float min_value, float max_value, int axis) {
		this.min_value = min_value;
		this.max_value = max_value;
		this.axis = axis;
	}
	public MinMaxNorm(float min_value, float max_value, float rate, int axis) {
		this.min_value = min_value;
		this.max_value = max_value;
		this.rate = rate;
		this.axis = axis;
	}
	@Override
	public void apply(INDArray m) {
		INDArray norm = Transforms.sqrt(Transforms.pow(m, 2.0f).sum(true, this.axis));
		INDArray desired = Nd4j.math().clipByValue(norm.dup(), this.min_value, this.max_value);
		if(rate != 1.0f) desired.muli(rate).addi(norm.mul(1.0f-rate));
		m.muli(desired).divi(norm.add(1e-7f));
	}
}
