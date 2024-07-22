package snickrs.ailibrary.autodiff.loss;

import snickrs.ailibrary.autodiff.*;

public class SquaredHingeNode extends UnaryNode {
	public SquaredHingeNode(Node target, Node prediction) {
		super(create(target, prediction));
	}
	public static Node create(Node target, Node prediction) {
		return new SumNode(new ExponentNode(new MaxNode(new SubtractionNode(new ConstantNode(1.0f), new ProductNode(target, prediction)), new ConstantNode(0.0f)), new ConstantNode(2.0f)));
	}
}
