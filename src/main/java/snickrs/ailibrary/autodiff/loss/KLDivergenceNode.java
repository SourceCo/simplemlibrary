package snickrs.ailibrary.autodiff.loss;

import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.math.*;

public class KLDivergenceNode extends UnaryNode {
	public KLDivergenceNode(Node target, Node prediction) {
		super(create(target, prediction));
	}
	public static Node create(Node target, Node prediction) {
		return new SumNode(new ProductNode(target, new LogNode(new QuotientNode(target, prediction))));
	}
}
