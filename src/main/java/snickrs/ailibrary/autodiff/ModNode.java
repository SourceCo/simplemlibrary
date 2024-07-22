package snickrs.ailibrary.autodiff;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class ModNode extends Node {
	public ModNode(Node left, Node right) {
		super(left, right);
	}
	public INDArray child_evaluate() {
		this.m = Nd4j.math().mod(children.get(0).evaluate().dup(), children.get(1).evaluate().dup());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, unbroadcast(children.get(0).m, upstream));
		partials.set(1, unbroadcast(children.get(1).m, upstream.neg().mul(Transforms.floor(children.get(0).m.div(children.get(1).m)))));
	}
    public long[] shape() {
    	return broadcast(children.get(0).shape(), children.get(1).shape());
    }
}