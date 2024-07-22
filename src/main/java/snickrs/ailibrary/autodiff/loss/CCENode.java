package snickrs.ailibrary.autodiff.loss;

import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.math.*;

public class CCENode extends UnaryNode {
	public CCENode(Node target, Node prediction) {
		super(create(target, prediction));
	}
	public static Node create(Node target, Node prediction) {
		Node norm = new QuotientNode(prediction, new SumNode(prediction));
		Node output = new ClipNode(norm, 1e-7f, 1.0f-1e-7f);
		return new NegNode(new SumNode(new ProductNode(target, new LogNode(output))));
	}
}
