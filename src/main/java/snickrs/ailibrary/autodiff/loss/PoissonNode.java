package snickrs.ailibrary.autodiff.loss;

import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.math.*;

public class PoissonNode extends UnaryNode {
	public PoissonNode(Node target, Node prediction) {
		super(create(target, prediction));
	}
	public static Node create(Node target, Node prediction) {
		return new SumNode(new SubtractionNode(prediction, new ProductNode(target, new LogNode(prediction))));
	}
}