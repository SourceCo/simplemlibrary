package snickrs.ailibrary.autodiff.optimization;

import java.util.Arrays;
import java.util.HashMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class Adam extends Optimizer {
	public float beta_1 = 0.9f; // often set to 0.9
	public float beta_2 = 0.999f; // often set to 0.999
	public float epsilon = 1e-7f; // small constant used to prevent division by 0
	public boolean amsgrad = false;
	HashMap<Variable, INDArray> moment1;
	HashMap<Variable, INDArray> moment2;
	HashMap<Variable, INDArray> max_moment2;
	public Adam(float lr, float beta_1, float beta_2, float weight_decay, float epsilon, boolean amsgrad) {
		super(lr, weight_decay);
		this.beta_1 = beta_1;
		this.beta_2 = beta_2;
		this.epsilon = epsilon;
		this.amsgrad = amsgrad;
		moment1 = new HashMap<Variable, INDArray>();
		moment2 = new HashMap<Variable, INDArray>();
		if(amsgrad) {
			max_moment2 = new HashMap<Variable, INDArray>();
		}
	}
	public void child_update(Variable variable) {
		if(moment1.get(variable) == null || !Arrays.equals(moment1.get(variable).shape(), variable.m.shape())) moment1.put(variable, Nd4j.zerosLike(variable.m));
		if(moment2.get(variable) == null || !Arrays.equals(moment2.get(variable).shape(), variable.m.shape())) moment2.put(variable, Nd4j.zerosLike(variable.m));
		if(amsgrad) {
			if(max_moment2.get(variable) == null || !Arrays.equals(max_moment2.get(variable).shape(), variable.m.shape())) max_moment2.put(variable, Nd4j.zerosLike(variable.m));
			moment1.get(variable).muli(Math.pow(beta_1, variable.updates+1.0f)).addi(variable.gradient.mul(1.0f-Math.pow(beta_1, variable.updates+1.0f)));
			moment2.get(variable).muli(beta_2).addi(Transforms.pow(variable.gradient, 2.0f).mul(1.0f-beta_2));
			max_moment2.put(variable, Transforms.max(max_moment2.get(variable), moment2.get(variable)));
			variable.m.subi(moment1.get(variable).mul(lr).div(Transforms.sqrt(max_moment2.get(variable)).add(epsilon)));
		} else {
			moment1.get(variable).muli(beta_1).addi(variable.gradient.mul(1.0f-beta_1));
			moment2.get(variable).muli(beta_2).addi(Transforms.pow(variable.gradient, 2.0f).mul(1.0f-beta_2));
			variable.m.subi(moment1.get(variable).mul(lr/(1.0f-Math.pow(beta_1, variable.updates+1.0f))).div(Transforms.sqrt(moment2.get(variable).mul(1.0f/(1.0f-Math.pow(beta_2, variable.updates+1.0f)))).add(epsilon)));
		}
	}
}
