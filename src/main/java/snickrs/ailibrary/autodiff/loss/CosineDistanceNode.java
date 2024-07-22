package snickrs.ailibrary.autodiff.loss;

import snickrs.ailibrary.autodiff.*;

public class CosineDistanceNode extends UnaryNode {
	public CosineDistanceNode(Node target, Node prediction) {
		super(create(target, prediction));
	}
	public static Node create(Node target, Node prediction) {
//		return new SumNode(new ProductNode(new L2NormNode(target), new L2NormNode(prediction)));
		return new SubtractionNode(new ConstantNode(1.0f), new QuotientNode(new SumNode(new ProductNode(target, prediction)), new ProductNode(new L2NormNode(target), new L2NormNode(prediction))));
	}
}
