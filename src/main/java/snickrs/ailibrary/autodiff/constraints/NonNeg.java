package snickrs.ailibrary.autodiff.constraints;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class NonNeg implements Constraint {
	public NonNeg() {
	}
	@Override
	public void apply(INDArray m) {
		Transforms.max(m, 0.0f, false);
	}
}
