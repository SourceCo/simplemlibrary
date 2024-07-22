package snickrs.ailibrary.autodiff.loss;

import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.math.*;

public class CFCENode extends UnaryNode {
	public CFCENode(Node target, Node prediction) {
		super(create(0.25f, 2.0f, target, prediction));
	}
	public CFCENode(float alpha, float gamma, Node target, Node prediction) {
		super(create(alpha, gamma, target, prediction));
	}
	public static Node create(float alpha, float gamma, Node target, Node prediction) {
//		return new NegNode(
//				new SumNode(
//				new AdditionNode(
//						new ProductNode(
//								new ProductNode(new ConstantNode(alpha), 
//										new ExponentNode(new SubtractionNode(new ConstantNode(1.0f), prediction), 
//										new ConstantNode(gamma))), 
//								new ProductNode(target, new LogNode(prediction))), 
//						new ProductNode(
//										new ProductNode(new ConstantNode(1.0f-alpha), new ExponentNode(prediction, new ConstantNode(gamma))), 
//										new ProductNode(new SubtractionNode(new ConstantNode(1.0f), target), new LogNode(new SubtractionNode(new ConstantNode(1.0f), prediction)))
//										))));
		Node output = new QuotientNode(prediction, new SumNode(prediction));
		Node cce = new NegNode(new ProductNode(target, new LogNode(output)));
		Node modulating_factor = new ExponentNode(new SubtractionNode(new ConstantNode(1.0f), output), new ConstantNode(gamma));
		Node weighting_factor = new ProductNode(modulating_factor, new ConstantNode(alpha));
		return new SumNode(new ProductNode(weighting_factor, cce));
//		return new NegNode(new SumNode(new ProductNode(new ConstantNode(alpha), new ExponentNode(new SubtractionNode(new ConstantNode(1.0f), prediction), new ConstantNode(gamma)), target, new LogNode(prediction))));
	}
}