package snickrs.ailibrary.autodiff;

import java.util.Arrays;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;

public class SubtractionNode extends Node {
	public SubtractionNode(Node ... children) {
		super(children[0], children.length <= 2 ? children[1] : new AdditionNode(Arrays.copyOfRange(children, 1, children.length)));
	}
	public SubtractionNode(List<Node> children) {
		super(children.get(0), children.size() <= 2 ? children.get(1) : new AdditionNode(children.subList(1, children.size())));
	}
	public INDArray child_evaluate() {
		this.m = children.get(0).evaluate().sub(children.get(1).evaluate());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, unbroadcast(children.get(0).m, upstream));
		partials.set(1, unbroadcast(children.get(1).m, upstream.neg()));
	}
    public long[] shape() {
    	return broadcast(children.get(0).shape(), children.get(1).shape());
    }
}
