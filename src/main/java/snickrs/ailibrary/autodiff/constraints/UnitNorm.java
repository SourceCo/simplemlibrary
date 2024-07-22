package snickrs.ailibrary.autodiff.constraints;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class UnitNorm implements Constraint {
	public int axis;
	public UnitNorm() {
		this.axis = 0;
	}
	public UnitNorm(int axis) {
		this.axis = axis;
	}
	@Override
	public void apply(INDArray m) {
		INDArray norm = Transforms.sqrt(Transforms.pow(m, 2.0f).sum(true, this.axis));
		m.divi(norm.add(1e-7f));
	}
}
