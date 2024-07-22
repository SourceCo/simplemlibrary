package snickrs.ailibrary.autodiff.optimization;

import java.util.Arrays;
import java.util.HashMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class RMSProp extends Optimizer {
	public float rho = 0.9f; // often set to 0.9
	public float epsilon = 1e-7f; // small constant used to prevent division by 0
	public HashMap<Variable, INDArray> gradsq;
	public RMSProp(float lr, float rho, float weight_decay, float epsilon) {
		super(lr, weight_decay);
		this.rho = rho;
		this.epsilon = epsilon;
		gradsq = new HashMap<Variable, INDArray>();
	}
	@Override
	public void child_update(Variable variable) {
		if(gradsq.get(variable) == null || !Arrays.equals(gradsq.get(variable).shape(), variable.m.shape())) gradsq.put(variable, Nd4j.zerosLike(variable.m));
		gradsq.get(variable).muli(rho).addi(Transforms.pow(variable.gradient, 2.0f).mul(1.0f-rho));
		variable.m.subi(variable.gradient.mul(lr).div(Transforms.sqrt(gradsq.get(variable).add(epsilon))));
	}
}
