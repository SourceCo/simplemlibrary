package snickrs.ailibrary.autodiff.loss;

import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.math.*;

public class BFCENode extends UnaryNode {
	public BFCENode(Node target, Node prediction) {
		super(create(0.25f, 2.0f, target, prediction));
	}
	public BFCENode(float alpha, float gamma, Node target, Node prediction) {
		super(create(alpha, gamma, target, prediction));
	}
	public static Node create(float alpha, float gamma, Node target, Node prediction) {
//		return new NegNode(new SumNode(new AdditionNode(
//				new ProductNode(
//						new ProductNode(new ConstantNode(alpha), target), 
//						new ProductNode(new LogNode(prediction), new ExponentNode(new SubtractionNode(new ConstantNode(1.0f), prediction), new ConstantNode(gamma)))
//						), 
//				new ProductNode(
//						new SubtractionNode(new ConstantNode(1.0f), target), 
//						new ProductNode(new ExponentNode(prediction, new ConstantNode(gamma)), new LogNode(new SubtractionNode(new ConstantNode(1.0f), prediction)))
//						)
//				)));
		Node p_t = new AdditionNode(new ProductNode(target, prediction), new ProductNode(new SubtractionNode(new ConstantNode(1.0f), target), new SubtractionNode(new ConstantNode(1.0f), prediction)));
		Node focal_factor = new ExponentNode(new SubtractionNode(new ConstantNode(1.0f), p_t), new ConstantNode(gamma));
		Node bce = new SumNode(new NegNode(new AdditionNode(new ProductNode(target, new LogNode(prediction)), new ProductNode(new SubtractionNode(new ConstantNode(1.0f), target), new LogNode(new SubtractionNode(new ConstantNode(1.0f), prediction)))))); // same as the BCENode
		return new SumNode(new ProductNode(focal_factor, bce));
//		return new NegNode(new SumNode(new AdditionNode(new ProductNode(new ConstantNode(alpha), target, new LogNode(prediction), new ExponentNode(new SubtractionNode(new ConstantNode(1.0f), prediction), new ConstantNode(gamma))), new ProductNode(new SubtractionNode(new ConstantNode(1.0f), target), new ExponentNode(prediction, new ConstantNode(gamma)), new LogNode(new SubtractionNode(new ConstantNode(1.0f), prediction))))));
	}
}
