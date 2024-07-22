package snickrs.ailibrary.autodiff.loss;

import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.math.*;

public class BCENode extends UnaryNode {
	public BCENode(Node target, Node prediction) {
		super(create(target, prediction));
	}
	public static Node create(Node target, Node prediction) {
		return new SumNode(new NegNode(new AdditionNode(new ProductNode(target, new LogNode(prediction)), new ProductNode(new SubtractionNode(new ConstantNode(1.0f), target), new LogNode(new SubtractionNode(new ConstantNode(1.0f), prediction))))));
	}
}
