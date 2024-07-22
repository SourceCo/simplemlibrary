package snickrs.ailibrary.autodiff.constraints;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Constraint {
	// note that constraints are typically only applied to weights
	public void apply(INDArray m);
}
