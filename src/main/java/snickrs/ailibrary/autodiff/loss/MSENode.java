package snickrs.ailibrary.autodiff.loss;

import snickrs.ailibrary.autodiff.*;

public class MSENode extends UnaryNode {
	public MSENode(Node target, Node prediction) {
		super(create(target, prediction));
	}
	public static Node create(Node target, Node prediction) {
		return new MeanNode(new ExponentNode(new SubtractionNode(target, prediction), new ConstantNode(2.0f)));
	}
}
