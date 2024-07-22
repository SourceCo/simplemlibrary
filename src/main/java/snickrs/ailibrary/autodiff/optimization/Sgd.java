package snickrs.ailibrary.autodiff.optimization;

import java.util.Arrays;
import java.util.HashMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import snickrs.ailibrary.autodiff.*;

public class Sgd extends Optimizer {
	public float momentum = 0.0f; // normally set between 0.8 and 0.99
	public boolean nesterov = false; // use the nesterov accelerated gradient optimization or not
	public HashMap<Variable, INDArray> velocities;
	public Sgd(float lr, float momentum, float weight_decay, boolean nesterov) {
		super(lr, weight_decay);
		this.momentum = momentum;
		this.nesterov = nesterov;
		velocities = new HashMap<Variable, INDArray>();
	}
	@Override
	public void child_update(Variable variable) {
		if(momentum != 0.0f) {
			if(velocities.get(variable) == null || !Arrays.equals(velocities.get(variable).shape(), variable.m.shape())) velocities.put(variable, Nd4j.zerosLike(variable.m));
			velocities.get(variable).muli(momentum).subi(variable.gradient.mul(lr));
			if(nesterov) {
				variable.m.addi(velocities.get(variable).mul(momentum)).sub(variable.gradient.mul(lr));
			} else {
				variable.m.addi(velocities.get(variable));
			}
		} else {
			variable.m.subi(variable.gradient.mul(lr));
		}
	}
}
