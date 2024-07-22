package snickrs.ailibrary.autodiff;

import java.util.Arrays;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class ExponentNode extends Node {
	public ExponentNode(Node ... children) {
		super(children[0], children.length <= 2 ? children[1] : new ProductNode(Arrays.copyOfRange(children, 1, children.length)));
	}
	public ExponentNode(List<Node> children) {
		super(children.get(0), children.size() <= 2 ? children.get(1) : new ProductNode(children.subList(1, children.size())));
	}
	public INDArray child_evaluate() {
		this.m = Transforms.pow(children.get(0).evaluate(), children.get(1).evaluate());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, unbroadcast(children.get(0).m, upstream.mul(children.get(1).m).mul(Transforms.pow(children.get(0).m, children.get(1).m.sub(1.0f)))));
		partials.set(1, unbroadcast(children.get(1).m, upstream.mul(Transforms.pow(children.get(0).m, children.get(1).m), Transforms.log(children.get(0).m))));
	}
    public long[] shape() {
    	return broadcast(children.get(0).shape(), children.get(1).shape());
    }
}
