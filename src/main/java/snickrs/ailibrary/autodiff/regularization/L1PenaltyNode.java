package snickrs.ailibrary.autodiff.regularization;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class L1PenaltyNode extends Node {
	public float coef;
	public L1PenaltyNode(Variable ... parameters) {
		super(parameters);
		this.coef = 0.001f;
	}
	public L1PenaltyNode(float coef, Variable ... parameters) {
		super(parameters);
		this.coef = coef;
	}
	public L1PenaltyNode(List<Variable> parameters) {
		super(parameters);
		this.coef = 0.001f;
	}
	public L1PenaltyNode(float coef, List<Variable> parameters) {
		super(parameters);
		this.coef = coef;
	}
	public INDArray child_evaluate() {
		float sum = 0.0f;
		for(int i = 0; i < children.size(); i++) {
			sum += Transforms.abs(children.get(i).m).sumNumber().floatValue();
		}
		this.m = Nd4j.scalar(coef*sum);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		for(int i = 0; i < children.size(); i++) {
			partials.set(i, unbroadcast(children.get(i).m, upstream.mul(Transforms.sign(children.get(i).m).mul(coef))));
		}
	}
}
