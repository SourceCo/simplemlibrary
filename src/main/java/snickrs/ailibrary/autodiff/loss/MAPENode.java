package snickrs.ailibrary.autodiff.loss;

import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.math.*;

public class MAPENode extends UnaryNode {
	public MAPENode(Node target, Node prediction) {
		super(create(target, prediction));
	}
	public static Node create(Node target, Node prediction) {
		return new ProductNode(new ConstantNode(100.0f), new MeanNode(new AbsNode(new QuotientNode(new SubtractionNode(target, prediction), target))));
	}
}
