package snickrs.ailibrary.autodiff;

import snickrs.ailibrary.autodiff.math.*;

public class StdNode extends UnaryNode {
	public StdNode(Node input, int[] axes) {
		super(create(input, axes));
	}
	public StdNode(Node input) {
		super(create(input, new int[] {-1}));
	}
	public static Node create(Node input, int[] axes) {
		return new SqrtNode(new MeanNode(new SubtractionNode(new ExponentNode(input, new ConstantNode(2.0f)), new ExponentNode(new MeanNode(input, axes), new ConstantNode(2.0f))), axes));
	}
}
