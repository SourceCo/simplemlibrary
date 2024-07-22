package snickrs.ailibrary.autodiff.math;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class LogNode extends UnaryNode {
	public static final float E = 2.7182818f;
	public float base;
	public LogNode(float base, Node left) {
		super(left);
		this.base = base;
	}
	public LogNode(Node left) {
		super(left);
		this.base = E; // natural log
	}
	public INDArray child_evaluate() {
		if(base != E) this.m = Transforms.log(children.get(0).evaluate(), base);
		else this.m = Transforms.log(children.get(0).evaluate().add(1e-7f));
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		if(base != E) partials.set(0, upstream.div(children.get(0).m.mul(Math.log(base))));
		else partials.set(0, upstream.div(children.get(0).m.add(1e-7f)));
	}
}
