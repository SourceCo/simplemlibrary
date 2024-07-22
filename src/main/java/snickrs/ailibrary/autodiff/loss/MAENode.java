package snickrs.ailibrary.autodiff.loss;

import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.math.*;

public class MAENode extends UnaryNode {
	public MAENode(Node target, Node prediction) {
		super(create(target, prediction));
	}
	public static Node create(Node target, Node prediction) {
		return new MeanNode(new AbsNode(new SubtractionNode(target, prediction)));
	}
}
