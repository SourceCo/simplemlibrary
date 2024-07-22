package snickrs.ailibrary.autodiff.constraints;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MaxNorm implements Constraint {
	public float max_value;
	public int axis;
	public MaxNorm() {
		this.max_value = 2.0f;
		this.axis = 0;
	}
	public MaxNorm(float max_value) {
		this.max_value = max_value;
		this.axis = 0;
	}
	public MaxNorm(float max_value, int axis) {
		this.max_value = max_value;
		this.axis = axis;
	}
	@Override
	public void apply(INDArray m) {
		INDArray norm = Transforms.sqrt(Transforms.pow(m, 2.0f).sum(true, this.axis));
		INDArray desired = Nd4j.math().clipByValue(norm.dup(), 0.0f, this.max_value);
		m.muli(desired).divi(norm.add(1e-7f));
	}
}
