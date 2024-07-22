package snickrs.ailibrary.autodiff.loss;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import snickrs.ailibrary.autodiff.*;

public class HuberNode extends Node {
	public INDArray difference;
	public float delta = 1.0f; // usually set to 1
	public HuberNode(Node left, Node right) {
		super(left, right);
		this.delta = 1.0f;
	}
	public HuberNode(float delta, Node left, Node right) {
		super(left, right);
		this.delta = delta;
	}
	public INDArray child_evaluate() {
		this.difference = children.get(0).evaluate().sub(children.get(1).evaluate());
	    this.m = Nd4j.zerosLike(difference);
		for(int i = 0; i < m.length(); i++) {
			float x = this.difference.getFloat(i);
			this.m.putScalar(i, Math.abs(x) <= delta ? 0.5f*Math.pow(x, 2.0f) : delta*(Math.abs(x)-0.5f*delta));
		}
	    this.m = this.m.mean(-1);
	    return this.m;
	}
	public void child_diff(INDArray upstream) {
		INDArray deriv = Nd4j.zerosLike(this.difference);
		for(int i = 0; i < difference.length(); i++) {
			float x = this.difference.getFloat(i);
			deriv.putScalar(i, Math.abs(x) <= delta ? x : delta*Math.signum(x));
		}
		deriv = upstream.mul(deriv);
		partials.set(0, deriv);
		partials.set(1, deriv.neg());
	}
}
