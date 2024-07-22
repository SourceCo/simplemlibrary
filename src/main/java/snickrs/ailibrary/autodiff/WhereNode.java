package snickrs.ailibrary.autodiff;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class WhereNode extends Node {
	public Node condition;
	public WhereNode(Node condition, Node left, Node right) {
		super(left, right);
		this.condition = condition;
	}
	public INDArray child_evaluate() {
		this.m = Nd4j.where(condition.evaluate(), children.get(0).evaluate(), children.get(1).evaluate())[0];
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		INDArray dL = Nd4j.zerosLike(this.m);
		INDArray dR = Nd4j.zerosLike(this.m);
		for(int i = 0; i < this.m.length(); i++) {
			float x = condition.m.getFloat(i);
			if(x == 1.0f) dL.putScalar(i, 1.0f);
			else dR.putScalar(i, 1.0f);
		}
		partials.set(0, unbroadcast(children.get(0).m, upstream.mul(dL)));
		partials.set(1, unbroadcast(children.get(1).m, upstream.mul(dR)));
	}
    public long[] shape() {
    	return broadcast(children.get(0).shape(), children.get(1).shape());
    }
}
